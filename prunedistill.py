#!/usr/bin/env python
"""
This script implements a pruning-distillation hybrid algorithm on LLMs.
(Pruning while incorporating KL divergence loss with the original model before pruning)

Usage:

CUDA_LAUNCH_BLOCKING=1 python3 prunedistill.py \
--model_name NousResearch/Llama-3.2-1B \
--output_dir "/afs/csail.mit.edu/u/a/asher/narrow/experiments/prunedistill" \
--regularizer "lhalf_of_l2_of_mlps" \
--lr 5e-5 \
--sparsity_lambda 5e-9 \
--max_steps 100000 \
--batch_size 8 \
--use_streaming \
--save_steps 20000 \
--do_sft
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["WANDB_PROJECT"] = "pruning"
from os.path import abspath, join, dirname

cache_dir = abspath(join(dirname(__file__), "..", ".cache", "huggingface"))
os.makedirs(cache_dir, exist_ok=True)

os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir

import argparse
import torch
import torch.nn as nn
from typing import Callable
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
import einops
import stat
import sys


# from transformer_lens.utils import ActivationCache
import torch.nn.utils.prune as prune
from typing import Dict, Union, Any, Mapping
import torch.nn.utils.prune as prune
import transformers
import wandb
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import copy
from huggingface_hub import HfFolder
from huggingface_hub.commands.user import login


class SparsityTrainer(Trainer):
    """
    Custom Trainer that adds a sparsity-inducing regularization term (e.g., L1)
    to the loss for causal language modeling, and incorporates a distillation loss
    based on a frozen teacher model's logits.
    """

    def __init__(
        self,
        *args,
        compute_sparsity_loss: Callable[[nn.Module], torch.Tensor],
        sparsity_lambda: float = 0.0,
        teacher_model: nn.Module = None,
        kl_weight: float = 1000.0,
        temperature: float = 2.0,
        target_kl: float = 1.3,
        kl_alpha_adjust_rate: float = 0.05,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.compute_sparsity_loss = compute_sparsity_loss
        self.sparsity_lambda = sparsity_lambda
        self.teacher_model = teacher_model
        self.kl_weight = kl_weight
        self.temperature = temperature
        self.target_kl = target_kl
        self.kl_alpha_adjust_rate = kl_alpha_adjust_rate
        self.weighted_kl_running_avg = None
        self.ema_beta = 0.9
        # Simplified KL weight adjustment parameters
        self.kl_threshold = 0.1  # Maximum allowed unweighted KL
        self.kl_weight_increase_factor = 1.01  # Very small increase (1%) when needed
        self.kl_ema = None  # Exponential moving average of unweighted KL
        self.kl_ema_decay = 0.99  # Decay factor for EMA (higher = more smoothing)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Check for updated hyperparameters from wandb
        if wandb.run is not None:
            self.sparsity_lambda = wandb.config.get(
                "sparsity_lambda", self.sparsity_lambda
            )

        try:
            # Compute the standard data loss and capture model outputs for logits.
            data_loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, **kwargs
            )
        except RuntimeError as e:
            print(
                f"Runtime error at step {self.state.global_step if hasattr(self, 'state') else 'unknown'}: {e}"
            )
            # Return a fallback loss and outputs
            dummy_loss = torch.tensor(1.0, device=next(model.parameters()).device)
            if return_outputs:
                # Create dummy outputs with the expected structure
                outputs = type("DummyOutputs", (), {})()
                outputs.logits = torch.zeros(
                    (
                        inputs["input_ids"].shape[0],
                        inputs["input_ids"].shape[1],
                        model.config.vocab_size,
                    ),
                    device=dummy_loss.device,
                )
                outputs.loss = dummy_loss
                return dummy_loss, outputs
            return dummy_loss

        # If a teacher model is provided, compute the distillation loss.
        if self.teacher_model is not None:
            try:
                self.teacher_model.eval()
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**inputs)
                    teacher_logits = teacher_outputs.logits.clone()  # Force a copy
                student_logits = outputs.logits
                T = self.temperature

                # Clip logits BEFORE temperature scaling to prevent extreme values
                max_val = 20.0  # More conservative clipping
                student_logits = torch.clamp(student_logits, -max_val, max_val)
                teacher_logits = torch.clamp(teacher_logits, -max_val, max_val)

                # Apply temperature scaling
                scaled_student_logits = student_logits / T
                scaled_teacher_logits = teacher_logits / T

                # KL calculation
                kd_loss = F.kl_div(
                    F.log_softmax(scaled_student_logits, dim=-1),
                    F.softmax(scaled_teacher_logits, dim=-1),
                    reduction="none",
                    log_target=False,
                ).sum(-1)

                # Average over non-padding tokens if attention mask is available
                if "attention_mask" in inputs:
                    mask = inputs["attention_mask"].float()
                    kd_loss = (kd_loss * mask).sum() / mask.sum().clamp(min=1.0)
                else:
                    kd_loss = kd_loss.mean()

                # Apply temperature scaling factor (T²)
                kd_loss = kd_loss * (T * T)

                # Get the unweighted KL loss for threshold checking
                unweighted_kl = kd_loss.item()

                # Update exponential moving average of unweighted KL
                if self.kl_ema is None:
                    self.kl_ema = unweighted_kl
                else:
                    self.kl_ema = self.kl_ema * self.kl_ema_decay + unweighted_kl * (
                        1 - self.kl_ema_decay
                    )

                # Calculate weighted KL loss with current weight
                weighted_kl_loss = self.kl_weight * kd_loss

                # Track running average of weighted KL loss
                if self.weighted_kl_running_avg is None:
                    self.weighted_kl_running_avg = weighted_kl_loss.item()
                else:
                    self.weighted_kl_running_avg = (
                        self.ema_beta * self.weighted_kl_running_avg
                        + (1 - self.ema_beta) * weighted_kl_loss.item()
                    )

                # First, adjust KL weight to target the desired weighted KL value
                kl_ratio = self.weighted_kl_running_avg / self.target_kl
                self.kl_weight *= 1 - self.kl_alpha_adjust_rate * (kl_ratio - 1)

                # Then, if unweighted KL is too high, increase weight regardless
                if self.kl_ema > self.kl_threshold:
                    self.kl_weight *= self.kl_weight_increase_factor
                    if wandb.run is not None:
                        wandb.log({"kl_weight_increased": 1.0})

                # Recalculate weighted KL loss with updated weight
                weighted_kl_loss = self.kl_weight * kd_loss

                # Combine data loss and KL loss
                combined_loss = data_loss + weighted_kl_loss

                # Create metrics dictionary for logging
                metrics = {
                    "unweighted_kl": unweighted_kl,
                    "kl_ema": self.kl_ema,
                    "weighted_kl_loss": weighted_kl_loss.item(),
                    "kl_weight": self.kl_weight,
                    "data_loss": data_loss.item(),
                }

                # Log metrics
                self.log(metrics)
                if wandb.run is not None:
                    wandb.log(metrics)

            except Exception as e:
                print(f"Error in KL calculation: {e}")
                # Fall back to just using the data loss
                combined_loss = data_loss
                if wandb.run is not None:
                    wandb.log({"kl_error": 1.0})
        else:
            raise Exception("No teacher model provided")

        try:
            # Calculate and add regularization loss
            reg_loss = self.sparsity_lambda * self.compute_sparsity_loss(model)

            # Check for NaN in regularization loss
            if torch.isnan(reg_loss):
                print(
                    "Warning: NaN detected in regularization loss, replacing with zero"
                )
                reg_loss = torch.tensor(0.0, device=combined_loss.device)

            total_loss = combined_loss + reg_loss
        except Exception as e:
            print(f"Error in regularization calculation: {e}")
            total_loss = combined_loss
            reg_loss = torch.tensor(0.0, device=combined_loss.device)

        # Final NaN check for total loss
        if torch.isnan(total_loss):
            print("Warning: NaN detected in total loss, falling back to data loss only")
            total_loss = data_loss

        # Final metrics to log
        final_metrics = {
            "reg_loss": (
                reg_loss.item() / self.sparsity_lambda
                if self.sparsity_lambda != 0
                else 0.0
            ),
            "reg_loss_weighted": reg_loss.item(),
            "combined_loss": combined_loss.item(),
            "total_loss": total_loss.item(),
            "sparsity_lambda": self.sparsity_lambda,
        }

        # Log to both trainer and wandb
        self.log(final_metrics)
        if wandb.run is not None:
            wandb.log(final_metrics)

        return (total_loss, outputs) if return_outputs else total_loss


def l1_sparsity_loss(model: nn.Module) -> torch.Tensor:
    """
    Compute the L1 norm of the model's trainable parameters.
    Potentially huge for large models—be mindful of scaling.
    """
    return sum(p.abs().sum() for p in model.parameters() if p.requires_grad)


def l1_sparsity_loss_mlps(model: nn.Module) -> torch.Tensor:
    """
    Compute the L1 norm of the model's MLP parameters (gate_proj, up_proj, down_proj).
    """
    device = next(model.parameters()).device
    loss = torch.tensor(0.0, device=device)

    for layer in model.model.layers:
        loss = loss + layer.mlp.gate_proj.weight.to(device).abs().sum()
        loss = loss + layer.mlp.up_proj.weight.to(device).abs().sum()
        loss = loss + layer.mlp.down_proj.weight.to(device).abs().sum()
    return loss


def l1_of_l2_of_mlps(model: nn.Module) -> torch.Tensor:
    """
    Computes the L1 norm of the L2 norm of the parameters specific to each MLP neuron.
    """
    ref_device = next(model.parameters()).device
    L1 = torch.tensor(0.0, device=ref_device)
    for layer in model.model.layers:
        gate_proj = layer.mlp.gate_proj.weight  # (4x, x)
        up_proj = layer.mlp.up_proj.weight  # (4x, x)
        down_proj = layer.mlp.down_proj.weight  # (x, 4x)
        L2 = torch.sqrt(
            gate_proj.pow(2).sum(dim=1)
            + up_proj.pow(2).sum(dim=1)
            + down_proj.pow(2).sum(dim=0)
        )
        L1 += L2.abs().sum().to(ref_device)
    return L1


def generalized_norm(
    model: nn.Module, outer_norm: float, inner_norm: float, eps: float = 1e-8
) -> torch.Tensor:
    """
    Computes the outer norm of the inner norm of the parameters specific to each MLP neuron.

    Computes, for each MLP layer:
       inner = ( sum(|gate_proj|^(inner_norm) over dim=1)
               + sum(|up_proj|^(inner_norm) over dim=1)
               + sum(|down_proj|^(inner_norm) over dim=0) + eps )^(1/inner_norm)

    And aggregates:
       aggregated_norm = ( Σ_neurons (inner^(outer_norm)) )^(1/outer_norm)

    Args:
        model (nn.Module): The model containing MLP layers (accessible via model.model.layers).
        outer_norm (float): The outer norm exponent for aggregation (e.g., 1/2, 1/5, etc.).
        inner_norm (float): The inner norm exponent for each neuron (e.g., 2 for L2 norm).
        eps (float, optional): A small constant added for numerical stability. Default is 1e-8.

    Returns:
        torch.Tensor: The computed aggregated norm.
    """
    # Get device from first parameter
    device = next(model.parameters()).device
    outer = torch.tensor(0.0, device=device)

    # Initialize list to collect all inner norms across layers
    all_inner_norms = []

    for layer in model.model.layers:
        # Ensure all tensors are on the correct device
        gate_proj = layer.mlp.gate_proj.weight.to(device)  # (4x, x)
        up_proj = layer.mlp.up_proj.weight.to(device)  # (4x, x)
        down_proj = layer.mlp.down_proj.weight.to(device)  # (x, 4x)

        inner = (
            gate_proj.pow(inner_norm).sum(dim=1)
            + up_proj.pow(inner_norm).sum(dim=1)
            + down_proj.pow(inner_norm).sum(dim=0)
            + eps
        ) ** (1.0 / inner_norm)

        # Collect the inner norms for this layer
        all_inner_norms.append(inner)

    # Concatenate all inner norms into a single tensor
    all_inner_norms = torch.cat(all_inner_norms)

    # Apply outer norm to the collected tensor
    return (all_inner_norms.abs().pow(outer_norm).sum() + eps) ** (1.0 / outer_norm)


REGULARIZERS = {
    "l1_sparsity_loss": l1_sparsity_loss,
    "l1_sparsity_loss_mlps": l1_sparsity_loss_mlps,
    "l1_of_l2_of_mlps": l1_of_l2_of_mlps,
    "lhalf_of_l2_of_mlps": lambda model: generalized_norm(model, 1 / 2, 2),
    "lfifth_of_l2_of_mlps": lambda model: generalized_norm(model, 1 / 5, 5),
    "lhalf_of_l1_of_mlps": lambda model: generalized_norm(model, 1 / 2, 1),
}


class PruneByAttributionCallback(TrainerCallback):
    def __init__(
        self,
        model: nn.Module,
        train_dataloader,
        prune_every_k_steps: int,
        importance_threshold: float = 1e-7,
        attribution_batch_size: int = 4,
    ):
        print(f"Initializing PruneByAttributionCallback with:")
        print(f"- Pruning every {prune_every_k_steps} steps")
        print(f"- Importance threshold: {importance_threshold}")
        print(f"- Attribution batch size: {attribution_batch_size}")
        self.model = model
        self.train_dataloader = train_dataloader
        self.prune_every_k_steps = prune_every_k_steps
        self.importance_threshold = importance_threshold
        self.attribution_batch_size = attribution_batch_size
        self.neuron_to_avg_effect = {}
        self.device = next(model.parameters()).device

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (state.global_step + 1) % self.prune_every_k_steps == 0:
            print(f"\nStep {state.global_step + 1}: Starting pruning cycle")
            self.neuron_to_avg_effect = {}
            print("Computing neuron importances...")
            self.compute_neuron_importances()

            # Compute and print attribution statistics across neurons
            import numpy as np

            print("Attribution statistics across neurons:")
            for cache_name, effects in self.neuron_to_avg_effect.items():
                values = list(effects.values())
                if values:
                    arr = np.array(values)
                    mean_attr = np.mean(arr)
                    percentile_10 = np.percentile(arr, 10)
                    percentile_20 = np.percentile(arr, 20)
                    percentile_30 = np.percentile(arr, 30)
                    print(
                        f"Stats for {cache_name}: Mean Attribution: {mean_attr:.6f}, "
                        f"10th Percentile: {percentile_10:.6f}, 20th Percentile: {percentile_20:.6f}, "
                        f"30th Percentile: {percentile_30:.6f}"
                    )

            print("Pruning neurons...")
            self.prune_neurons(self.model)
            print("Pruning complete\n")

    def compute_neuron_importances(self):
        print(f"Processing {self.attribution_batch_size} batches for attribution...")
        for batch_idx, inputs in enumerate(self.train_dataloader):
            print(f"Processing batch {batch_idx + 1}/{self.attribution_batch_size}")
            if batch_idx >= self.attribution_batch_size:
                break
            inputs = self._prepare_inputs(inputs)

            # Compute the effect of each neuron on the loss
            _, cache, grad_cache = self.get_cache_fwd_and_bwd(self.model, inputs)

            for layer_idx, layer in enumerate(self.model.model.layers):
                for mat_name in ["gate_proj", "up_proj", "down_proj"]:
                    matrix = getattr(layer.mlp, mat_name)
                    cache_key = id(matrix)

                    neuron_acts = cache.cache_dict[cache_key]
                    neuron_grads = grad_cache.cache_dict[cache_key]

                    for neuron_idx in range(matrix.weight.shape[0]):
                        # takes element-wise product between neuron gradients and activations, sums over sequence dimension
                        neuron_effect = einops.einsum(
                            neuron_grads[:, :, neuron_idx],
                            neuron_acts[:, :, neuron_idx],
                            "batch seq, batch seq -> batch",
                        )
                        avg_neuron_effect = neuron_effect.mean().item()

                        cache_name = f"layer_{layer_idx}_{mat_name}"
                        if cache_name not in self.neuron_to_avg_effect:
                            self.neuron_to_avg_effect[cache_name] = {}

                        if neuron_idx not in self.neuron_to_avg_effect[cache_name]:
                            self.neuron_to_avg_effect[cache_name][neuron_idx] = 0.0

                        self.neuron_to_avg_effect[cache_name][
                            neuron_idx
                        ] += avg_neuron_effect

        # Average the neuron effects over the batches
        for cache_name in self.neuron_to_avg_effect:
            for neuron_idx in self.neuron_to_avg_effect[cache_name]:
                self.neuron_to_avg_effect[cache_name][
                    neuron_idx
                ] /= self.attribution_batch_size

    def prune_neurons(self, model):
        total_neurons_pruned = 0
        total_neurons = 0

        for layer_idx in range(len(model.model.layers)):
            layer = model.model.layers[layer_idx]
            for mat_name in ["gate_proj", "up_proj", "down_proj"]:
                matrix = getattr(layer.mlp, mat_name)
                total_neurons += matrix.weight.shape[0]
                cache_name = f"layer_{layer_idx}_{mat_name}"

                neuron_effects = self.neuron_to_avg_effect[cache_name]

                # Prune neurons with absolute effect below threshold
                neurons_to_prune = [
                    neuron
                    for neuron, effect in neuron_effects.items()
                    if abs(effect) < self.importance_threshold
                ]

                # Create output dimension mask
                output_mask = torch.ones(
                    matrix.weight.shape[0], device=matrix.weight.device
                )
                output_mask[neurons_to_prune] = 0

                num_pruned = (output_mask == 0).sum().item()
                total_neurons_pruned += num_pruned

                full_mask = output_mask.unsqueeze(1).expand_as(matrix.weight)
                prune.custom_from_mask(matrix, name="weight", mask=full_mask)

            print(f"Layer {layer_idx}:")
            print(
                f"- Pruning {num_pruned} out of {layer.mlp.gate_proj.weight.shape[0]} neurons"
            )
        print(
            f"Total neurons pruned across all layers: {total_neurons_pruned} / {total_neurons} ({total_neurons_pruned/total_neurons:.2%})"
        )

    def _prepare_inputs(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        if isinstance(inputs, Mapping):
            return type(inputs)({k: self._prepare_input(v) for k, v in inputs.items()})

    def _prepare_input(
        self, data: Union[torch.Tensor, Any]
    ) -> Union[torch.Tensor, Any]:
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            return data.to(**kwargs)
        return data

    def get_cache_fwd_and_bwd(self, model, tokens):
        cache = {}
        grad_cache = {}

        def forward_hook(module, input, output):
            cache[id(module)] = output.detach()
            return output

        def backward_hook(module, grad_input, grad_output):
            grad_cache[id(module)] = grad_output[0].detach()
            return grad_input

        # Register hooks for MLP layers
        for name, module in model.named_modules():
            if "mlp" in name:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)

        # Forward and backward pass
        outputs = model(**tokens)
        loss = outputs.loss
        loss.backward()

        class SimpleCache:
            def __init__(self, cache_dict):
                self.cache_dict = cache_dict

            def __getitem__(self, key):
                return self.cache_dict[key]

        return (
            loss.item(),
            SimpleCache(cache),
            SimpleCache(grad_cache),
        )


class PruneByWeightNorm(TrainerCallback):
    def __init__(
        self,
        model: nn.Module,
        prune_every_k_steps: int,
        pruning_threshold: float,
    ):
        print(f"Initializing PruneByWeightNorm with:")
        print(f"- Pruning every {prune_every_k_steps} steps")
        print(f"- Pruning threshold: {pruning_threshold}")
        self.model = model
        self.prune_every_k_steps = prune_every_k_steps
        self.pruning_threshold = pruning_threshold

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (state.global_step + 1) % self.prune_every_k_steps == 0:
            print(f"\nStep {state.global_step + 1}: Starting weight norm pruning")
            self.prune_neurons(self.model, state.global_step)
            print("Weight norm pruning complete\n")

    def prune_neurons(self, model: nn.Module, global_step: int):
        total_neurons_pruned = 0
        total_neurons = 0
        overall_L2_list = []

        for i, layer in enumerate(model.model.layers):
            gate_proj = layer.mlp.gate_proj
            up_proj = layer.mlp.up_proj
            down_proj = layer.mlp.down_proj

            neurons_in_layer = gate_proj.weight.shape[0]
            total_neurons += neurons_in_layer

            L2 = torch.sqrt(
                gate_proj.weight.pow(2).sum(dim=1)
                + up_proj.weight.pow(2).sum(dim=1)
                + down_proj.weight.pow(2).sum(dim=0)
            )
            overall_L2_list.append(L2.detach().cpu())

            # Ensure the q tensor is on the same device and has same dtype as L2.
            q_tensor = torch.tensor([0.25, 0.5, 0.75], dtype=L2.dtype, device=L2.device)
            layer_quartiles = torch.quantile(L2, q_tensor).tolist()

            # Create a mask for neurons with L2 norm above (or equal) to the pruning threshold.
            output_mask = L2 >= self.pruning_threshold
            num_pruned = (~output_mask).sum().item()
            total_neurons_pruned += num_pruned

            print(f"\nLayer {i}:")
            print(f"  Pruning threshold: {self.pruning_threshold:.4f}")
            print(
                f"  Fraction pruned: {num_pruned} / {neurons_in_layer} = {num_pruned/neurons_in_layer:.2%}"
            )
            print(
                f"  L2 quartiles: 25th: {layer_quartiles[0]:.4f}, Median: {layer_quartiles[1]:.4f}, "
                f"75th: {layer_quartiles[2]:.4f}"
            )

            # Apply pruning based on the computed mask
            mask = output_mask.unsqueeze(1).expand_as(gate_proj.weight)

            # Update existing pruning if already applied; otherwise, apply the pruning.
            if hasattr(gate_proj, "weight_orig"):
                gate_proj.weight_mask.data.copy_(mask)
            else:
                prune.custom_from_mask(gate_proj, name="weight", mask=mask)
            if hasattr(up_proj, "weight_orig"):
                up_proj.weight_mask.data.copy_(mask)
            else:
                prune.custom_from_mask(up_proj, name="weight", mask=mask)
            if hasattr(down_proj, "weight_orig"):
                down_proj.weight_mask.data.copy_(mask.T)
            else:
                prune.custom_from_mask(down_proj, name="weight", mask=mask.T)

        # Aggregate overall statistics across layers.
        total_pruned_fraction = (
            total_neurons_pruned / total_neurons if total_neurons > 0 else 0
        )
        overall_L2 = torch.cat(overall_L2_list)
        # List of desired percentiles to log.
        percentiles_to_log = [5, 10, 25, 50]
        global_percentiles = {}
        for p in percentiles_to_log:
            q_value = torch.quantile(overall_L2, p / 100.0).item()
            global_percentiles[f"global_neuron_norm_{p}th_percentile"] = q_value

        print(
            f"\nTotal neurons pruned across all layers: {total_neurons_pruned} / {total_neurons} "
            f"({total_neurons_pruned/total_neurons:.2%})"
        )
        print("Global neuron norm percentiles:")
        for key, value in global_percentiles.items():
            print(f"  {key}: {value:.4f}")

        # Log the global percentiles to wandb (associated with the current global step).
        wandb.log(global_percentiles)


class SparsityLambdaScheduler(TrainerCallback):
    """
    A callback to decay the trainer's sparsity_lambda parameter by a factor of 5 at step 6000.
    """

    def __init__(self, trainer, decay_step: int = 6000, decay_factor: float = 0.17):
        self.trainer = trainer
        self.decay_step = decay_step
        self.decay_factor = decay_factor
        self.has_decayed = False

    def on_step_end(self, args, state, control, **kwargs):
        if (state.global_step + 1) == self.decay_step and not self.has_decayed:
            old_lambda = self.trainer.sparsity_lambda
            new_lambda = old_lambda * self.decay_factor
            self.trainer.sparsity_lambda = new_lambda
            self.trainer.log({"sparsity_lambda": new_lambda})
            print(
                f"Step {state.global_step + 1}: Decayed sparsity_lambda from {old_lambda:.6f} to {new_lambda:.6f}"
            )
            self.has_decayed = True
        return control


class LearningRateDecayCallback(TrainerCallback):
    def __init__(
        self,
        start_lr: float,
        final_lr: float,
        total_steps: int,
        warmup_steps: int = 1000,
    ):
        self.start_lr = start_lr
        self.final_lr = final_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.trainer = None  # Will be set after the callback is added to the trainer

    def on_step_end(self, args, state, control, **kwargs):
        # First handle warmup phase
        if state.global_step < self.warmup_steps:
            # Linear warmup from 1% of start_lr to full start_lr
            warmup_progress = state.global_step / self.warmup_steps
            computed_lr = (
                self.start_lr * 0.01 + (self.start_lr * 0.99) * warmup_progress
            )
        else:
            # After warmup, compute progress and update the learning rate linearly
            progress = min(
                (state.global_step - self.warmup_steps)
                / (self.total_steps - self.warmup_steps),
                1.0,
            )
            computed_lr = self.start_lr * (1 - progress) + self.final_lr * progress

        # If wandb is running, allow UI override of the computed learning rate
        if wandb.run is not None:
            computed_lr = wandb.config.get("learning_rate", computed_lr)

        # Use the callback's stored trainer reference if available, otherwise try from kwargs.
        trainer = (
            self.trainer if self.trainer is not None else kwargs.get("trainer", None)
        )
        if trainer is not None:
            for param_group in trainer.optimizer.param_groups:
                param_group["lr"] = computed_lr
            # Log the learning rate using the trainer's logging interface ...
            trainer.log({"learning_rate": computed_lr})
            # ... and also directly to wandb.
            wandb.log({"learning_rate": computed_lr, "step": state.global_step})
        return control


class ResetNetworkNormCallback(TrainerCallback):
    """
    A callback that resets the network's global norm after each training step.
    It does this by recording the global norm at the beginning of the step and,
    after the optimizer update, computing the factor by which the update scaled down
    the norm. It then multiplies every trainable parameter by that inverse factor.
    """

    def __init__(self):
        self.prev_global_norm = None

    def get_global_norm(self, model: torch.nn.Module) -> float:
        total_norm_sq = 0.0
        for param in model.parameters():
            if param.requires_grad:
                total_norm_sq += param.data.pow(2).sum().item()
        return total_norm_sq**0.5

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model") or kwargs.get("trainer").model
        self.prev_global_norm = self.get_global_norm(model)

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model") or kwargs.get("trainer").model
        current_global_norm = self.get_global_norm(model)
        if current_global_norm == 0:
            return control

        scaling_factor = self.prev_global_norm / current_global_norm

        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.data.mul_(scaling_factor)

        print(
            f"Step {state.global_step}: Network norm rescaled by factor {scaling_factor:.6f}"
        )
        return control


def parse_args():
    parser = argparse.ArgumentParser(description="Prune-finetuning script")
    parser.add_argument(
        "--model_name",
        type=str,
        default="NousResearch/Llama-3.2-1B",
        help="Model name or path to a pretrained model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./pruning_output",
        help="Directory to store model checkpoints and logs.",
    )
    parser.add_argument(
        "--sparsity_lambda",
        type=float,
        default=0.01,
        help="Regularization strength for L1 penalty.",
    )
    parser.add_argument(
        "--regularizer",
        type=str,
        default="l1_sparsity_loss_mlps",
        choices=REGULARIZERS.keys(),
        help="Regularization function to use.",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="Learning rate for Adam optimizer."
    )
    parser.add_argument(
        "--final_lr",
        type=float,
        default=2e-12,
        help="Final learning rate at end of training.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100000,
        help="Total number of training steps to run.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of total epochs to train (if not using max_steps).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Per-device batch size."
    )
    parser.add_argument(
        "--eval_steps", type=int, default=500, help="Perform evaluation every N steps."
    )
    parser.add_argument(
        "--logging_steps", type=int, default=5, help="Log every N steps."
    )
    parser.add_argument(
        "--save_steps", type=int, default=100000, help="Save checkpoint every N steps."
    )
    parser.add_argument(
        "--accumulations",
        type=int,
        default=2,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--use_streaming", action="store_true", help="Use streaming dataset if set."
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Number of warmup steps for learning rate scheduler.",
    )
    parser.add_argument(
        "--do_sft", action="store_true", help="Run SFT before pruning-distillation."
    )

    args = parser.parse_args()

    # Add debug info and better error handling for directory creation
    print(f"Attempting to create output directory: {args.output_dir}")
    print(f"Current user: {os.getuid()}")
    print(f"Current working directory: {os.getcwd()}")

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        # Test write permissions by creating a small test file
        test_file = os.path.join(args.output_dir, "test_permissions.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print(f"Successfully created and verified write access to: {args.output_dir}")
    except OSError as e:
        print(f"Error creating/accessing output directory: {e}")
        print("Directory details:")
        try:
            if os.path.exists(args.output_dir):
                print(f"Directory permissions: {oct(os.stat(args.output_dir).st_mode)}")
                print(f"Directory owner: {os.stat(args.output_dir).st_uid}")
        except OSError as e2:
            print(f"Could not get directory details: {e2}")

        print("\nPlease try one of the following:")
        print(
            "1. Use an absolute path in a writable directory (e.g., /tmp/pruning_output)"
        )
        print("2. Check and fix permissions on the target directory")
        print("3. Run the script with appropriate permissions")
        sys.exit(1)

    return args


def prepare_dataset(dataset, tokenizer, max_length):
    """
    Tokenize the dataset for causal language modeling.
    """

    def tokenize_function(examples):
        return tokenizer(
            examples["code"],
            truncation=True,
            max_length=max_length,
        )

    # For non-streaming dataset, we can .map() directly.
    # For streaming, we may have to do things differently.
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return tokenized_dataset


def get_exponential_warmup_scheduler(optimizer, num_warmup_steps, base_lr):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return base_lr ** (current_step / num_warmup_steps)
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def run_sft(model, tokenizer, train_dataset, val_dataset, args):
    """
    Performs Supervised Fine-Tuning (SFT) on the model to adapt it to the target distribution
    before pruning and distillation. For SFT, we use a gentler learning rate scheduler
    with an increased warmup period so that the learning rate ramps up very gradually,
    which helps prevent an initial loss spike. The SFT model is not saved to disk.
    """
    print("\n=== Starting Supervised Fine-Tuning (SFT) Phase ===\n")

    # Updated SFT hyperparameters for smoother loss:
    sft_epochs = 1
    sft_lr = 1e-6  # Lowered learning rate for gentler updates
    sft_batch_size = 8
    sft_warmup_steps = 2000  # Increased warmup steps for a more gradual ramp-up
    sft_max_steps = 10000  # HERE

    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # SFT output directory (used for logging; we won't save the final model)
    sft_output_dir = os.path.join(args.output_dir, "sft_model")
    os.makedirs(sft_output_dir, exist_ok=True)

    sft_training_args = TrainingArguments(
        output_dir=sft_output_dir,
        num_train_epochs=sft_epochs,
        max_steps=sft_max_steps,  # Explicit max_steps required for streaming datasets
        per_device_train_batch_size=sft_batch_size,
        per_device_eval_batch_size=sft_batch_size,
        logging_dir=os.path.join(sft_output_dir, "logs"),
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=args.eval_steps,
        save_strategy="no",  # Disable model saving during SFT
        optim="adamw_torch_fused",
        bf16=True,
        save_total_limit=2,
        learning_rate=sft_lr,
        warmup_steps=sft_warmup_steps,
        lr_scheduler_type="constant_with_warmup",  # Gentle ramp-up
        load_best_model_at_end=False,
        save_only_model=True,
        max_grad_norm=1.0,
        hub_token=None,
        save_safetensors=True,
    )

    # Update wandb config for SFT and allow overwriting values
    sft_config = {
        "phase": "sft",
        "learning_rate": sft_lr,
        "batch_size": sft_batch_size,
        "epochs": sft_epochs,
    }
    wandb.config.update(sft_config, allow_val_change=True)

    # Initialize a Trainer (without any regularization) for SFT
    sft_trainer = Trainer(
        model=model,
        args=sft_training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Run SFT
    sft_trainer.train()

    print("SFT phase completed. Using the fine-tuned model for pruning-distillation.\n")
    # Return the fine-tuned model directly without saving it to disk.
    return sft_trainer.model


def main():
    args = parse_args()
    print("args: ", args)

    # Check and handle Hugging Face login
    token = HfFolder.get_token()
    if token is None:
        print(
            "No Hugging Face token found. Please enter your token (from https://huggingface.co/settings/tokens):"
        )
        token = input().strip()
        login(token=token)
        if not HfFolder.get_token():
            raise ValueError(
                "Failed to login to Hugging Face. Please check your token and try again."
            )
    print("Successfully authenticated with Hugging Face")

    # Initialize wandb with config
    wandb_config = {
        "learning_rate": args.lr,
        "sparsity_lambda": args.sparsity_lambda,
        "warmup_steps": args.warmup_steps,
    }
    wandb.init(config=wandb_config, allow_val_change=True)

    # Generate a model ID based on wandb run name and config
    hub_model_id = f"pruned-{args.model_name.split('/')[-1]}-{wandb.run.name}"
    hub_model_id = hub_model_id.replace("/", "_").lower()

    # Load or stream dataset
    if args.use_streaming:
        print("Loading streaming dataset")
        dataset = load_dataset(
            "codeparrot/github-code",
            streaming=True,
            languages=["Python"],
            split="train",
            cache_dir="../.cache/huggingface/datasets",
        )
        train_dataset = dataset
        val_dataset = None
        attribution_dataset = load_dataset(
            "codeparrot/github-code",
            streaming=True,
            languages=["Python"],
            split="train",
            cache_dir="../.cache/huggingface/datasets",
        )
    else:
        print("Loading non-streaming dataset")
        dataset = load_dataset(
            "codeparrot/github-code", languages=["Python"], split="train"
        )
        train_dataset = dataset.select(range(0, int(0.8 * len(dataset))))
        val_dataset = dataset.select(range(int(0.8 * len(dataset)), len(dataset)))
        attribution_dataset = dataset.select(range(1000))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_train = prepare_dataset(train_dataset, tokenizer, args.max_length)
    tokenized_val = (
        prepare_dataset(val_dataset, tokenizer, args.max_length)
        if val_dataset
        else None
    )

    print("loading model")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    print("finished loading model")

    # Run SFT if enabled
    if args.do_sft:
        model = run_sft(model, tokenizer, tokenized_train, tokenized_val, args)

    # Reinitialize wandb for the pruning-distillation phase
    wandb.config.update({"phase": "pruning-distillation"}, allow_val_change=True)

    # Create a teacher model from the fine-tuned model
    if args.do_sft:
        # Option 1: Create a more explicit deep copy
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        # Load the SFT model state dict into the teacher
        teacher_model.load_state_dict(model.state_dict())

        # Option 2: Alternative approach - save and reload the model
        # temp_path = os.path.join(args.output_dir, "temp_sft_model")
        # model.save_pretrained(temp_path)
        # teacher_model = AutoModelForCausalLM.from_pretrained(
        #     temp_path,
        #     torch_dtype=torch.float32,
        #     device_map="auto",
        # )
    else:
        teacher_model = copy.deepcopy(model)

    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulations,
        per_device_eval_batch_size=args.batch_size,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        optim="adamw_torch_fused",
        bf16=True,
        gradient_checkpointing=False,
        save_total_limit=2,
        learning_rate=args.lr,
        warmup_steps=0,
        lr_scheduler_type="constant",
        load_best_model_at_end=False,
        save_only_model=True,
        max_grad_norm=1.0,
        save_safetensors=True,
        # Add Hub-specific arguments
        push_to_hub=True,
        hub_model_id=hub_model_id,
        hub_strategy="every_save",
    )
    training_args.save_optimizer_state = False

    attribution_tokenized = prepare_dataset(
        attribution_dataset, tokenizer, args.max_length
    )
    attribution_dataloader = torch.utils.data.DataLoader(
        attribution_tokenized,
        batch_size=args.batch_size,
        collate_fn=data_collator,
    )

    trainer = SparsityTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_sparsity_loss=REGULARIZERS[args.regularizer],
        sparsity_lambda=args.sparsity_lambda,
        teacher_model=teacher_model,
        kl_weight=wandb.config.get("distillation_alpha", 100.0),
        temperature=2.0,
        target_kl=wandb.config.get("target_kl", 1.0),
        kl_alpha_adjust_rate=wandb.config.get("kl_alpha_adjust_rate", 0.01),
        callbacks=[
            LearningRateDecayCallback(
                start_lr=args.lr,
                final_lr=args.final_lr,
                total_steps=training_args.max_steps,
                warmup_steps=args.warmup_steps,
            ),
        ],
    )

    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, (SparsityLambdaScheduler, LearningRateDecayCallback)):
            callback.trainer = trainer

    def save_checkpoint():
        try:
            print(f"Pushing final model to the Hub as {hub_model_id}")

            # Create a model card with training details
            model_card = f"""
            # Pruned {args.model_name}
            
            This model was created by applying pruning-distillation to {args.model_name}.
            
            ## Training Parameters
            - Regularizer: {args.regularizer}
            - Sparsity Lambda: {args.sparsity_lambda}
            - Learning Rate: {args.lr}
            - Batch Size: {args.batch_size}
            - Training Steps: {args.max_steps}
            
            ## Wandb Run
            - Run Name: {wandb.run.name}
            - Run URL: {wandb.run.url}
            """

            # Save model card
            with open(os.path.join(args.output_dir, "README.md"), "w") as f:
                f.write(model_card)

            trainer.push_to_hub(
                repo_id=hub_model_id,
                commit_message="Final pruned model",
            )
            print("Successfully pushed model to the Hub")
        except Exception as e:
            print(f"Error pushing to Hub: {e}")
            print("Falling back to local save...")
            checkpoint_dir = os.path.join(args.output_dir, "final_model")
            trainer.save_model(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            return False
        return True

    trainer.train()
    save_checkpoint()


if __name__ == "__main__":
    main()
