#!/usr/bin/env python
"""
This script implements prune-finetuning with group lasso style regularization.

Usage:
    python tuneprune.py \
    --output_dir=/afs/csail.mit.edu/u/a/asher/narrow \
    --regularizer=lhalf_of_l2_of_mlps \
    --lr=2e-6 \
    --sparsity_lambda=1.5e-8 \
    --max_steps=100000 \
    --batch_size=32 \
    --use_streaming \
    --save_steps=50000

"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
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
from torch.optim.optimizer import Optimizer


class SparsityTrainer(Trainer):
    """
    Custom Trainer that adds a sparsity-inducing regularization term (e.g., L1)
    to the loss for causal language modeling.
    """

    def __init__(
        self,
        *args,
        compute_sparsity_loss: Callable[[nn.Module], torch.Tensor],
        sparsity_lambda: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.compute_sparsity_loss = compute_sparsity_loss
        self.sparsity_lambda = sparsity_lambda

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Check for updated hyperparameters from wandb
        if wandb.run is not None:
            self.sparsity_lambda = wandb.config.get(
                "sparsity_lambda", self.sparsity_lambda
            )

        data_loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, **kwargs
        )
        reg_loss = self.sparsity_lambda * self.compute_sparsity_loss(model)
        total_loss = data_loss + reg_loss
        self.log({"reg_loss": reg_loss.item() / self.sparsity_lambda})
        self.log({"reg_loss_weighted": reg_loss.item()})
        self.log({"data_loss": data_loss.item()})
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
    def __init__(self, start_lr: float, final_lr: float, total_steps: int):
        self.start_lr = start_lr
        self.final_lr = final_lr
        self.total_steps = total_steps
        self.trainer = None  # Will be set after the callback is added to the trainer

    def on_step_end(self, args, state, control, **kwargs):
        # Compute progress and update the learning rate linearly.
        progress = min(state.global_step / self.total_steps, 1.0)
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


class SNRAdam(Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, alpha=1.0, weight_decay=0
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps, alpha=alpha, weight_decay=weight_decay
        )
        super(SNRAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("SNRAdam does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                # Compute SNR
                snr = corrected_exp_avg.pow(2) / (corrected_exp_avg_sq + group["eps"])

                # Compute the update
                update = corrected_exp_avg / (
                    corrected_exp_avg_sq.sqrt() + group["eps"]
                )
                update.mul_(snr.pow(group["alpha"]))

                if group["weight_decay"] != 0:
                    update.add_(p.data, alpha=group["weight_decay"])

                # Apply the update
                p.data.add_(update, alpha=-group["lr"])

        return loss


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
        default=10000,
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
        "--save_steps", type=int, default=10000, help="Save checkpoint every N steps."
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


def main():
    args = parse_args()
    print("args: ", args)

    # Initialize wandb with config
    wandb_config = {
        "learning_rate": args.lr,
        "sparsity_lambda": args.sparsity_lambda,
    }
    wandb.init(config=wandb_config, allow_val_change=True)

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
        # Streaming datasets typically don't have "train" / "validation" splits
        # or random access. One might need to do something like:
        #   train_dataset = dataset.take(100000)
        #   val_dataset   = dataset.skip(100000).take(20000)
        # Adjust accordingly for your use case.
        # For simplicity, let's pretend we just do a single dataset:
        # train_dataset = dataset.take(1000)
        train_dataset = dataset
        val_dataset = None

        # For streaming dataset, create attribution dataset from a new stream
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
        )  # For demonstration
        # Create a small validation split just as an example
        train_dataset = dataset.select(range(0, int(0.8 * len(dataset))))
        val_dataset = dataset.select(range(int(0.8 * len(dataset)), len(dataset)))
        attribution_dataset = dataset.select(range(1000))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        # Ensure we have a pad token, especially important for GPT-2-like models
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
        torch_dtype=torch.float32,  # or float16 if your GPU supports it
        device_map="auto",  # could also specify device like "cuda:0"
    )
    print("finished loading model")
    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize the custom optimizer
    optimizer = SNRAdam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        alpha=1.0,
        weight_decay=0,
    )

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
        bf16=True,  # BF16
        gradient_checkpointing=False,  # Potential memory savings
        save_total_limit=2,  # EDITS: Only keep the latest two checkpoints
        learning_rate=args.lr,
        warmup_steps=0,  # Disable built-in warmup (and decay) to allow our callback to control lr.
        lr_scheduler_type="constant",  # Use a constant scheduler so our LR callback can override it.
        load_best_model_at_end=False,  # Prevent saving optimizer state
        save_only_model=True,  # If using newer transformers versions
        max_grad_norm=1.0,
        hub_token=None,  # Disable model hub interactions
        save_safetensors=True,  # Use safetensors format which is more robust
    )
    # EDITS: Disable saving optimizer state to save space.
    training_args.save_optimizer_state = False

    # Create dataloader for attribution
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
        optimizers=(optimizer, None),  # Pass the custom optimizer here
        # callbacks=[
        #     SparsityLambdaScheduler(trainer=None, decay_step=6000, decay_factor=0.13),
        #     LearningRateDecayCallback(
        #         start_lr=args.lr,
        #         final_lr=args.final_lr,
        #         total_steps=training_args.max_steps,
        #     ),
        # ],
    )

    # After initializing the trainer, update the callback's trainer reference.
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, (SparsityLambdaScheduler, LearningRateDecayCallback)):
            callback.trainer = trainer

    # Add error handling around model saving
    def save_checkpoint():
        try:
            checkpoint_dir = os.path.join(args.output_dir, "final_model")
            print(f"Attempting to save model to: {checkpoint_dir}")
            trainer.save_model(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print("Successfully saved model and tokenizer")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            print("Try running with a different output directory or check permissions")
            return False
        return True

    trainer.train()
    save_checkpoint()


if __name__ == "__main__":
    main()
