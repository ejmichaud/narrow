#!/usr/bin/env python
"""
This script implements prune-finetuning by:
1) Subclassing the Hugging Face Trainer to allow a loss function that
   incentivizes structural sparsity in the model (e.g., L1 regularization).
2) Training a causal language model with an additional L1 penalty on its weights.
3) Saving the final model checkpoints.

Usage:
    python prune_finetune.py --model_name <model> --num_train_epochs <epochs> ...
"""
import os
from os.path import abspath, join, dirname

cache_dir = abspath(join(dirname(__file__), "..", ".cache", "huggingface"))
os.makedirs(cache_dir, exist_ok=True)

os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"

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

# from transformer_lens.utils import ActivationCache
import torch.nn.utils.prune as prune
from typing import Dict, Union, Any, Mapping
import torch.nn.utils.prune as prune
import transformers


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


def lhalf_of_l2_of_mlps(model: nn.Module) -> torch.Tensor:
    """
    Computes the L1/2 norm of the L2 norm of the parameters specific to each MLP neuron.
    """
    # Get device from first parameter
    device = next(model.parameters()).device
    Lhalf = torch.tensor(0.0, device=device)

    for layer in model.model.layers:
        # Ensure all tensors are on the same device
        gate_proj = layer.mlp.gate_proj.weight.to(device)  # (4x, x)
        up_proj = layer.mlp.up_proj.weight.to(device)  # (4x, x)
        down_proj = layer.mlp.down_proj.weight.to(device)  # (x, 4x)

        L2 = torch.sqrt(
            gate_proj.pow(2).sum(dim=1)
            + up_proj.pow(2).sum(dim=1)
            + down_proj.pow(2).sum(dim=0)
        )
        Lhalf += L2.abs().pow(0.5).sum()
    return Lhalf.pow(2)


REGULARIZERS = {
    "l1_sparsity_loss": l1_sparsity_loss,
    "l1_sparsity_loss_mlps": l1_sparsity_loss_mlps,
    "l1_of_l2_of_mlps": l1_of_l2_of_mlps,
    "lhalf_of_l2_of_mlps": lhalf_of_l2_of_mlps,
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
            self.prune_neurons(self.model)
            print("Weight norm pruning complete\n")

    def prune_neurons(self, model):
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
        q_tensor_global = torch.tensor(
            [0.25, 0.5, 0.75], dtype=overall_L2.dtype, device=overall_L2.device
        )
        global_quartiles = torch.quantile(overall_L2, q_tensor_global).tolist()

        print(
            f"\nTotal neurons pruned across all layers: {total_neurons_pruned} / {total_neurons} "
            f"({total_pruned_fraction:.2%})"
        )
        print(
            "Global L2 quartiles across all neurons: "
            f"25th: {global_quartiles[0]:.4f}, Median: {global_quartiles[1]:.4f}, 75th: {global_quartiles[2]:.4f}\n"
        )


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
        "--accumulations",
        type=int,
        default=2,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--eval_steps", type=int, default=500, help="Perform evaluation every N steps."
    )
    parser.add_argument(
        "--logging_steps", type=int, default=5, help="Log every N steps."
    )
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Save checkpoint every N steps."
    )
    parser.add_argument(
        "--use_streaming", action="store_true", help="Use streaming dataset if set."
    )
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
        "--use_streaming", action="store_true", help="Use streaming dataset if set."
    )
    return parser.parse_args()


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


def main():
    args = parse_args()
    print("args: ", args)

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

    # Define training arguments
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
        optim="adamw_torch_fused",  # FASTER OPTIMIZER
        # fp16=True,  # Set to True if you want mixed precision (and your GPU supports it)
        bf16=True,  # BF16
        gradient_checkpointing=False,  # Potential memory savings
        save_total_limit=2,  # EDITS: Only keep the latest two checkpoints
        gradient_accumulation_steps=1,
        fp16=False,  # Set to True if you want mixed precision (and your GPU supports it)
        gradient_checkpointing=True,  # Potential memory savings
        learning_rate=args.lr,
        warmup_steps=1000,
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
        # callbacks=[
        #     PruneByAttributionCallback(
        #         model=model,
        #         train_dataloader=attribution_dataloader,
        #         prune_every_k_steps=10000,
        #         importance_threshold=1e-7,
        #         attribution_batch_size=args.batch_size,
        #     )
        # ],
        callbacks=[
            PruneByWeightNorm(
                model=model,
                prune_every_k_steps=100,
                pruning_threshold=0.2,
            )
        ],
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))

    # You might also want to save the tokenizer
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))


if __name__ == "__main__":
    main()
