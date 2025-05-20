#!/usr/bin/env python3
"""
Prune a transformer model using one of two strategies (“attribution” or “weight_norm”)
and evaluate the pruned model on a test dataset. Evaluation statistics are saved
to disk.
"""

import argparse
import json
import os
os.environ['HF_HOME'] = '/om/user/ericjm/.cache/huggingface'
from collections.abc import Mapping
from typing import Any, Dict, Tuple, Union

from tqdm.auto import tqdm
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
    streaming: bool = True,
    skip_samples: int = 0,
) -> DataLoader:
    """
    Load and tokenize a dataset for pruning or evaluation.

    If the dataset is streamed, you can optionally skip a number of documents,
    allowing you to use one part of the stream for attribution and a different part for evaluation.
    
    Args:
        dataset_name: Name of the dataset to load.
        model_name: Name of the model used to load its tokenizer.
        max_length: Maximum token length.
        batch_size: Batch size to use in the DataLoader.
        num_samples: Number of samples to use from the dataset.
        split: Which split of the dataset to use (e.g. "train", "test").
        streaming: Whether to load the dataset in streaming mode.
        skip_samples: Number of samples to skip from the beginning of the stream.
    
    Returns:
        A DataLoader yielding batches suitable for language modeling.
    """
    if streaming:
        dataset = load_dataset(dataset_name, split=split, languages=['Python'], streaming=True)
        if skip_samples > 0:
            dataset = dataset.skip(skip_samples)
        dataset = dataset.take(num_samples)
        # Convert streaming dataset to a list to allow tokenization.
        dataset = list(dataset)
    else:
        dataset = load_dataset(dataset_name, split=split, languages=['Python'])
        # For non-streaming, we assume random access is available.
        dataset = dataset.select(range(skip_samples, skip_samples + num_samples))

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

    tokenized_dataset = (
        dataset
        if isinstance(dataset, list)
        else dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    )
    # If tokenized_dataset is a list (from streaming), tokenize each sample.
    if isinstance(tokenized_dataset, list):
        tokenized_dataset = [tokenize_function(sample) for sample in tokenized_dataset]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)
    return dataloader


@torch.no_grad()
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
    for batch in tqdm(dataloader, desc="evaluating..."):
        batch = move_to_device(batch, device)
        outputs = model(**batch)
        losses.append(outputs.loss.item())
    return {
        "mean_loss": np.mean(losses).item(),
        "std_of_mean": (np.std(losses) / np.sqrt(len(losses))).item(),
        "losses": losses,
    }


def prune_by_attribution(
    model: nn.Module,
    train_dataloader: DataLoader,
    importance_threshold: float,
    num_attribution_batches: int,
    save_pruned_model: bool,
    output_dir: str, 
):
    """Prune neurons based on their attribution scores."""

    def get_attribution_hook(cache, name, hook_cache):
        def attribution_hook(module, input, output):
            def backward_hook(grad):
                modified_grad = -output.detach() * grad
                cache[name] = modified_grad
                return grad
            hook_cache[name] = output.register_hook(backward_hook)
            return None 
        return attribution_hook

    scores = {layeri: 0 for layeri in range(len(model.model.layers))}
    total_activations = {layeri: 0 for layeri in range(len(model.model.layers))}
    for i, batch in enumerate(tqdm(train_dataloader, desc="attribution scores...")):
        if i >= num_attribution_batches:
            break
        
        cache = {}
        forward_hooks = {}
        backward_handles = {}
        for layeri in range(len(model.model.layers)):
            forward_hooks[layeri] = model.model.layers[layeri].mlp.act_fn.register_forward_hook(
                get_attribution_hook(cache, layeri, backward_handles)
            )

        batch = move_to_device(batch, model.device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # remove hooks and sum attribution scores across batch and seq dim
        for layeri in range(len(model.model.layers)):
            attrs = cache[layeri]
            scores[layeri] += attrs.sum(dim=tuple(range(attrs.ndim - 1))).detach()
            total_activations[layeri] += attrs.shape[0] * attrs.shape[1]
            forward_hooks[layeri].remove()
            backward_handles[layeri].remove()
        
        del cache
        del forward_hooks
        del backward_handles

    # average scores
    for layeri in scores:
        scores[layeri] /= total_activations[layeri]
    
    # prune by score
    total_neurons_pruned = 0
    total_neurons = 0
    neurons_pruned_per_layer = []
    for layeri, layer in enumerate(model.model.layers):
        mask = scores[layeri].abs() < importance_threshold
        total_neurons += mask.shape[0]
        total_neurons_pruned += mask.sum().item()
        neurons_pruned_per_layer.append(mask.sum().item())
        print(f"Layer {layeri}: pruned {mask.sum().item()} out of {mask.shape[0]} neurons")
        # custom pruning implementation:
        for neuroni in range(mask.shape[0]):
            if mask[neuroni]:
                layer.mlp.gate_proj.weight.data[neuroni] = 0.0
                layer.mlp.up_proj.weight.data[neuroni] = 0.0
                layer.mlp.down_proj.weight.data[:, neuroni] = 0.0

    print(
        f"Total neurons pruned: {total_neurons_pruned} / {total_neurons} "
        f"({total_neurons_pruned/total_neurons:.2%})"
    )

    if save_pruned_model:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        print(f"Pruned model saved to {output_dir}")

    return total_neurons_pruned, total_neurons, neurons_pruned_per_layer


def prune_by_weight_norm(
    model: nn.Module,
    pruning_threshold: float,
    save_pruned_model: bool,
    output_dir: str,
) -> Tuple[int, int]:
    """
    Prune neurons by comparing the L2 norm of their associated weights to a threshold.
    
    Returns:
        A tuple (total_neurons_pruned, total_neurons) across all layers.
    """
    total_neurons_pruned = 0
    total_neurons = 0
    neurons_pruned_per_layer = []

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

        neurons_pruned_per_layer.append(num_pruned)
        print(f"Layer {layer_idx}: pruned {num_pruned} out of {len(output_mask)} neurons")

    print(
        f"Total neurons pruned: {total_neurons_pruned} / {total_neurons} "
        f"({total_neurons_pruned/total_neurons:.2%})"
    )

    if save_pruned_model:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        print(f"Pruned model saved to {output_dir}")

    return total_neurons_pruned, total_neurons, neurons_pruned_per_layer


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
    parser.add_argument("--test_split", type=str, default="train",
                        help="Dataset split to use for evaluation.")
    parser.add_argument("--num_test_samples", type=int, default=200,
                        help="Number of samples to use for evaluation.")
    # New argument: how many samples to skip for evaluation (if using the same split).
    parser.add_argument("--eval_skip", type=int, default=0,
                        help="For streaming datasets, number of samples to skip for evaluation.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    assert args.num_samples % args.batch_size == 0, "num_samples must be divisible by batch_size"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model.
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map=str(device)
    )

    # Apply pruning strategy and capture pruning statistics.
    if args.pruning_strategy == "attribution":

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
            skip_samples=0,  # For attribution, start at the beginning.
        )

        pruning_stats = prune_by_attribution(
            model=model,
            train_dataloader=pruning_dataloader,
            importance_threshold=args.importance_threshold,
            num_attribution_batches=args.num_samples // args.batch_size,
            save_pruned_model=args.save_pruned_model,
            output_dir=args.output_dir,
        )
    elif args.pruning_strategy == "weight_norm":
        pruning_stats = prune_by_weight_norm(
            model=model,
            pruning_threshold=args.pruning_threshold,
            save_pruned_model=args.save_pruned_model,
            output_dir=args.output_dir,
        )
    else:
        print("Invalid pruning strategy selected.")
        return

    # Load evaluation data.
    # If the dataset only has one split, you can still stream it and skip over the documents used for attribution.
    print("Preparing evaluation data...")
    test_dataloader = prepare_data(
        dataset_name=args.test_dataset_name,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_samples=args.num_test_samples,
        split=args.test_split,
        streaming=args.streaming,  # Streaming can be used for evaluation as well.
        skip_samples=args.eval_skip,  # Skip over documents already used for attribution.
    )

    print("Evaluating pruned model...")
    eval_stats = evaluate_model(model, test_dataloader, device)
    # Log pruning statistics with evaluation results.
    eval_stats["pruning_stats"] = {
        "total_neurons_pruned": pruning_stats[0],
        "total_neurons": pruning_stats[1],
        "neurons_pruned_per_layer": pruning_stats[2],
    }

    # Save evaluation statistics.
    os.makedirs(args.output_dir, exist_ok=True)
    eval_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(eval_file, "w") as f:
        json.dump(eval_stats, f, indent=4)
    print(f"Evaluation statistics saved to {eval_file}")


if __name__ == "__main__":
    main()
