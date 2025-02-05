#!/usr/bin/env python3
"""
Prune a transformer model using one of two strategies (“attribution” or “weight_norm”)
and evaluate the pruned model on a test dataset. Evaluation statistics are saved
to disk.
"""

import argparse
import json
import os
from collections.abc import Mapping
from typing import Any, Dict, Union

import numpy as np
import einops
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling


def move_to_device(data: Any, device: torch.device) -> Any:
    """
    Recursively move tensors (or containers of tensors) to the specified device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, Mapping):
        return type(data)({k: move_to_device(v, device) for k, v in data.items()})
    if isinstance(data, (list, tuple)):
        return type(data)(move_to_device(v, device) for v in data)
    return data


def prepare_data(
    dataset_name: str,
    model_name: str,
    max_length: int,
    batch_size: int,
    num_samples: int,
    split: str = "train",
    streaming: bool = False,
) -> DataLoader:
    """
    Load and tokenize a dataset for pruning or evaluation.
    
    Args:
        dataset_name: Name of the dataset to load.
        model_name: Name of the model used to load its tokenizer.
        max_length: Maximum token length.
        batch_size: Batch size to use in the DataLoader.
        num_samples: Number of samples to use from the dataset.
        split: Which split of the dataset to use (e.g. "train", "test").
        streaming: Whether to load the dataset in streaming mode.
    
    Returns:
        A DataLoader yielding batches suitable for language modeling.
    """
    if streaming:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        dataset = dataset.take(num_samples)
        # Convert streaming dataset to a list to allow tokenization.
        dataset = list(dataset)
    else:
        dataset = load_dataset(dataset_name, split=split)
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        # Assumes the field "code" holds the text; adjust as necessary.
        return tokenizer(
            examples["code"],
            truncation=True,
            max_length=max_length,
        )

    tokenized_dataset = dataset if isinstance(dataset, list) else dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    # If tokenized_dataset is a list (from streaming), tokenize each sample.
    if isinstance(tokenized_dataset, list):
        tokenized_dataset = [tokenize_function(sample) for sample in tokenized_dataset]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)
    return dataloader


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Evaluate the model on the provided data and compute average loss.
    
    Args:
        model: The language model to evaluate.
        dataloader: DataLoader providing evaluation batches.
        device: The device on which the model and data reside.
    
    Returns:
        A dictionary of evaluation statistics.
    """
    model.eval()
    losses = []
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = move_to_device(batch, device)
            outputs = model(**batch)
            losses.append(outputs.loss.item())
            # Multiply loss by number of tokens in the batch.
            # batch_size = batch["input_ids"].size(0)
            # seq_length = batch["input_ids"].size(1)
            # total_loss += outputs.loss.item() * batch_size * seq_length
            # total_tokens += batch_size * seq_length

    # avg_loss = total_loss / total_tokens if total_tokens > 0 else float("nan")
    # return {"average_loss": avg_loss}
    # return mean loss, std of mean, and list of losses
    return {
        "mean_loss": np.mean(losses).item(),
        "std_of_mean": np.std(losses).item() / np.sqrt(len(losses)).item(),
        "losses": losses,
    }


def prune_by_attribution(
    model: nn.Module,
    train_dataloader: DataLoader,
    importance_threshold: float,
    num_attribution_batches: int,
    save_pruned_model: bool,
    output_dir: str,
) -> None:
    """
    Prune neurons based on their contribution (attribution) to the loss.
    
    For each of a fixed number of batches from the dataloader, the forward and
    backward activations for each neuron are cached. Their average effect is
    computed and neurons whose effect is below a threshold are pruned.
    """
    device = next(model.parameters()).device
    neuron_to_avg_effect: Dict[str, Dict[int, float]] = {}

    def _prepare_inputs(inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        return {k: move_to_device(v, device) for k, v in inputs.items()}

    def get_cache_fwd_and_bwd(model: nn.Module, tokens: Dict[str, torch.Tensor]):
        cache = {}
        grad_cache = {}

        def forward_hook(module, input, output):
            cache[id(module)] = output.detach()
            return output

        def backward_hook(module, grad_input, grad_output):
            # Store only the gradient for the first output.
            grad_cache[id(module)] = grad_output[0].detach()
            return grad_input

        # Register hooks only for modules with "mlp" in their name.
        hooks = []
        for name, module in model.named_modules():
            if "mlp" in name:
                hooks.append(module.register_forward_hook(forward_hook))
                hooks.append(module.register_full_backward_hook(backward_hook))

        outputs = model(**tokens)
        loss = outputs.loss
        loss.backward()

        # Remove hooks.
        for hook in hooks:
            hook.remove()

        # Wrap caches in a simple object.
        class SimpleCache:
            def __init__(self, cache_dict):
                self.cache_dict = cache_dict

            def __getitem__(self, key):
                return self.cache_dict[key]

        return loss.item(), SimpleCache(cache), SimpleCache(grad_cache)

    def compute_neuron_importances() -> None:
        print(
            f"Computing neuron importances from {num_attribution_batches} batches..."
        )
        batch_count = 0
        for inputs in train_dataloader:
            if batch_count >= num_attribution_batches:
                break
            print(f"Processing attribution batch {batch_count + 1}/{num_attribution_batches}")
            inputs = _prepare_inputs(inputs)
            _, cache, grad_cache = get_cache_fwd_and_bwd(model, inputs)

            # Iterate over layers of the transformer model.
            # (Assumes the model has an attribute "model.layers" containing the layers.)
            for layer_idx, layer in enumerate(model.model.layers):
                for mat_name in ["gate_proj", "up_proj", "down_proj"]:
                    matrix = getattr(layer.mlp, mat_name)
                    cache_key = id(matrix)

                    neuron_acts = cache.cache_dict.get(cache_key)
                    neuron_grads = grad_cache.cache_dict.get(cache_key)
                    if neuron_acts is None or neuron_grads is None:
                        continue

                    cache_name = f"layer_{layer_idx}_{mat_name}"
                    if cache_name not in neuron_to_avg_effect:
                        neuron_to_avg_effect[cache_name] = {}

                    # For each neuron (row of the weight matrix)
                    for neuron_idx in range(matrix.weight.shape[0]):
                        # Compute the element-wise product between activations and gradients,
                        # then sum over the sequence dimension.
                        neuron_effect = einops.einsum(
                            neuron_grads[:, :, neuron_idx],
                            neuron_acts[:, :, neuron_idx],
                            "batch seq, batch seq -> batch",
                        )
                        avg_effect = neuron_effect.mean().item()
                        neuron_to_avg_effect[cache_name][neuron_idx] = (
                            neuron_to_avg_effect[cache_name].get(neuron_idx, 0.0) + avg_effect
                        )
            batch_count += 1

        # Average the effects over the number of batches.
        for cache_name in neuron_to_avg_effect:
            for neuron_idx in neuron_to_avg_effect[cache_name]:
                neuron_to_avg_effect[cache_name][neuron_idx] /= num_attribution_batches

    def prune_neurons() -> None:
        total_neurons_pruned = 0
        total_neurons = 0

        for layer_idx, layer in enumerate(model.model.layers):
            for mat_name in ["gate_proj", "up_proj", "down_proj"]:
                matrix = getattr(layer.mlp, mat_name)
                total_neurons += matrix.weight.shape[0]
                cache_name = f"layer_{layer_idx}_{mat_name}"
                neuron_effects = neuron_to_avg_effect.get(cache_name, {})
                # Identify neurons with absolute average effect below the threshold.
                neurons_to_prune = [
                    neuron_idx
                    for neuron_idx, effect in neuron_effects.items()
                    if abs(effect) < importance_threshold
                ]
                # Create a binary mask (1 = keep, 0 = prune)
                output_mask = torch.ones(matrix.weight.shape[0], device=matrix.weight.device)
                output_mask[neurons_to_prune] = 0

                num_pruned = int((output_mask == 0).sum().item())
                total_neurons_pruned += num_pruned

                full_mask = output_mask.unsqueeze(1).expand_as(matrix.weight)
                prune.custom_from_mask(matrix, name="weight", mask=full_mask)

            print(f"Layer {layer_idx}: pruned neurons in projections (last num_pruned={num_pruned}).")
        print(
            f"Total neurons pruned: {total_neurons_pruned} / {total_neurons} "
            f"({total_neurons_pruned/total_neurons:.2%})"
        )

    # Run attribution-based pruning.
    compute_neuron_importances()
    prune_neurons()

    if save_pruned_model:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        print(f"Pruned model saved to {output_dir}")


def prune_by_weight_norm(
    model: nn.Module,
    pruning_threshold: float,
    save_pruned_model: bool,
    output_dir: str,
) -> None:
    """
    Prune neurons by comparing the L2 norm of their associated weights to a threshold.
    """
    total_neurons_pruned = 0
    total_neurons = 0

    for layer_idx, layer in enumerate(model.model.layers):
        gate_proj = layer.mlp.gate_proj
        up_proj = layer.mlp.up_proj
        down_proj = layer.mlp.down_proj

        total_neurons += gate_proj.weight.shape[0]
        # Compute an L2 norm that aggregates contributions from three matrices.
        L2 = torch.sqrt(
            gate_proj.weight.pow(2).sum(dim=1)
            + up_proj.weight.pow(2).sum(dim=1)
            + down_proj.weight.pow(2).sum(dim=0)
        )
        # Create a mask for neurons whose norm meets the threshold.
        output_mask = L2 >= pruning_threshold
        num_pruned = int((~output_mask).sum().item())
        total_neurons_pruned += num_pruned

        mask = output_mask.unsqueeze(1).expand_as(gate_proj.weight)
        prune.custom_from_mask(gate_proj, name="weight", mask=mask)
        prune.custom_from_mask(up_proj, name="weight", mask=mask)
        prune.custom_from_mask(down_proj, name="weight", mask=mask.T)

        print(f"Layer {layer_idx}: pruned {num_pruned} out of {len(output_mask)} neurons")

    print(
        f"Total neurons pruned: {total_neurons_pruned} / {total_neurons} "
        f"({total_neurons_pruned/total_neurons:.2%})"
    )

    if save_pruned_model:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        print(f"Pruned model saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prune a language model using attribution or weight norm and evaluate it."
    )
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-3.2-1B",
                        help="Pretrained model name or path.")
    parser.add_argument("--pruning_strategy", type=str, choices=["attribution", "weight_norm"],
                        default="attribution", help="Pruning strategy to use.")
    parser.add_argument("--dataset_name", type=str, default="codeparrot/github-code",
                        help="Dataset name for pruning (training data).")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for data loading.")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to use for pruning data.")
    parser.add_argument("--streaming", action="store_true",
                        help="Load the dataset in streaming mode.")

    # Attribution strategy parameters
    parser.add_argument("--attribution_batch_size", type=int, default=4,
                        help="Batch size used for attribution pruning (not used separately if using same dataloader).")
    parser.add_argument("--num_attribution_batches", type=int, default=4,
                        help="Number of batches to use for computing attributions.")
    parser.add_argument("--importance_threshold", type=float, default=1e-7,
                        help="Threshold for neuron importance (attribution pruning).")

    # Weight norm strategy parameter
    parser.add_argument("--pruning_threshold", type=float, default=1.5,
                        help="Threshold for neuron L2 norm (weight norm pruning).")

    parser.add_argument("--save_pruned_model", action="store_true",
                        help="Whether to save the pruned model.")
    parser.add_argument("--output_dir", type=str, default="./pruned_models",
                        help="Directory to save the pruned model and evaluation results.")

    # Evaluation parameters
    parser.add_argument("--test_dataset_name", type=str, default="codeparrot/github-code",
                        help="Dataset name for evaluation.")
    parser.add_argument("--test_split", type=str, default="test",
                        help="Dataset split to use for evaluation.")
    parser.add_argument("--num_test_samples", type=int, default=200,
                        help="Number of samples to use for evaluation.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model.
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map={-1: device},  # Ensure model is loaded to the correct device.
    )

    # Load pruning data.
    print("Preparing pruning data...")
    pruning_dataloader = prepare_data(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        split="train",
        streaming=args.streaming,
    )

    # Apply pruning strategy.
    if args.pruning_strategy == "attribution":
        prune_by_attribution(
            model=model,
            train_dataloader=pruning_dataloader,
            importance_threshold=args.importance_threshold,
            num_attribution_batches=args.num_attribution_batches,
            save_pruned_model=args.save_pruned_model,
            output_dir=args.output_dir,
        )
    elif args.pruning_strategy == "weight_norm":
        prune_by_weight_norm(
            model=model,
            pruning_threshold=args.pruning_threshold,
            save_pruned_model=args.save_pruned_model,
            output_dir=args.output_dir,
        )
    else:
        print("Invalid pruning strategy selected.")
        return

    # Evaluate the pruned model on test data.
    print("Preparing evaluation data...")
    test_dataloader = prepare_data(
        dataset_name=args.test_dataset_name,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_samples=args.num_test_samples,
        split=args.test_split,
        streaming=False,  # usually evaluation is done on a fixed set in memory
    )

    print("Evaluating pruned model...")
    eval_stats = evaluate_model(model, test_dataloader, device)
    print(f"Evaluation results: {eval_stats}")

    # Save evaluation statistics.
    os.makedirs(args.output_dir, exist_ok=True)
    eval_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(eval_file, "w") as f:
        json.dump(eval_stats, f, indent=4)
    print(f"Evaluation statistics saved to {eval_file}")


if __name__ == "__main__":
    main()
