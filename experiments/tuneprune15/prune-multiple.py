#!/usr/bin/env python3
"""
Prune a transformer model using an attribution-based strategy and evaluate the pruned model on a test dataset.
Attribution scores are computed once and then used to prune with different sparsity levels.
Evaluation statistics are saved to disk.
"""

import argparse
import json
import os
from collections.abc import Mapping
from collections import defaultdict
from typing import Any, Dict, Tuple

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
import copy


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
    Load and tokenize a dataset for attribution computation or evaluation.
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
        dataset = dataset.select(range(skip_samples, skip_samples + num_samples))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
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
    if isinstance(tokenized_dataset, list):
        tokenized_dataset = [tokenize_function(sample) for sample in tokenized_dataset]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)
    return dataloader


@torch.no_grad()
def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Evaluate the model on the provided data and compute average loss.
    """
    model.eval()
    losses = []
    for batch in tqdm(dataloader, desc="Evaluating..."):
        batch = move_to_device(batch, device)
        outputs = model(**batch)
        losses.append(outputs.loss.item())
    return {
        "mean_loss": np.mean(losses).item(),
        "std_of_mean": (np.std(losses) / np.sqrt(len(losses))).item(),
        "losses": losses,
    }

def compute_attribution_scores(model: nn.Module, train_dataloader: DataLoader, num_attribution_batches: int):
    """
    Compute attribution scores for neurons across the model's layers using a hook-based approach.
    Returns a sorted list of tuples: (layer index, neuron index, attribution score).
    """
    def get_attribution_hook(cache, name, hook_cache):
        def attribution_hook(module, input, output):
            def backward_hook(grad):
                modified_grad = -output.detach() * grad
                cache[name] = modified_grad
                return grad
            hook_cache[name] = output.register_hook(backward_hook)
            return None
        return attribution_hook

    # Get device from model
    device = next(model.parameters()).device
    
    num_layers = len(model.model.layers)
    scores = {layeri: 0 for layeri in range(num_layers)}
    total_activations = {layeri: 0 for layeri in range(num_layers)}

    for i, batch in enumerate(tqdm(train_dataloader, desc="Computing attribution scores...")):
        if i >= num_attribution_batches:
            break
        
        cache = {}
        forward_hooks = {}
        backward_handles = {}
        for layeri in range(num_layers):
            forward_hooks[layeri] = model.model.layers[layeri].mlp.act_fn.register_forward_hook(
                get_attribution_hook(cache, layeri, backward_handles)
            )

        batch = move_to_device(batch, device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        for layeri in range(num_layers):
            attrs = cache[layeri]
            scores[layeri] += attrs.sum(dim=tuple(range(attrs.ndim - 1))).detach().abs()
            total_activations[layeri] += attrs.shape[0] * attrs.shape[1]
            forward_hooks[layeri].remove()
            backward_handles[layeri].remove()
        
        del cache, forward_hooks, backward_handles

    # Average scores per neuron.
    for layeri in scores:
        scores[layeri] /= total_activations[layeri]

    score_tuples = []
    for layeri in range(num_layers):
        for neuron_idx in range(scores[layeri].shape[0]):
            score_tuples.append((layeri, neuron_idx, scores[layeri][neuron_idx].item()))
    score_tuples.sort(key=lambda x: x[2])
    return score_tuples


def prune_model_by_scores(model: nn.Module, scores: list, sparsity: float, save_pruned_model: bool, output_dir: str) -> Tuple[int, int, Dict[int, int]]:
    """
    Prune the lowest scoring neurons based on the provided attribution scores and sparsity level.
    """
    total_neurons = len(scores)
    num_pruned = int(sparsity * total_neurons)
    neurons_pruned_per_layer = defaultdict(int)
    total_neurons_pruned = 0

    for i in tqdm(range(num_pruned), desc="Pruning neurons..."):
        layer_idx, neuron_idx, _ = scores[i]
        layer = model.model.layers[layer_idx]
        gate_proj = layer.mlp.gate_proj
        up_proj = layer.mlp.up_proj
        down_proj = layer.mlp.down_proj
        gate_proj.weight.data[neuron_idx] = 0.0
        up_proj.weight.data[neuron_idx] = 0.0
        down_proj.weight.data[:, neuron_idx] = 0.0
        neurons_pruned_per_layer[layer_idx] += 1
        total_neurons_pruned += 1

    print(
        f"Total neurons pruned: {total_neurons_pruned} / {total_neurons} "
        f"({total_neurons_pruned/total_neurons:.2%})"
    )
    if save_pruned_model:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        print(f"Pruned model saved to {output_dir}")

    return total_neurons_pruned, total_neurons, dict(neurons_pruned_per_layer)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prune a language model using attribution pruning and evaluate it across different sparsities."
    )
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-3.2-1B",
                        help="Pretrained model name or path.")
    parser.add_argument("--dataset_name", type=str, default="codeparrot/github-code",
                        help="Dataset name for attribution computation (training data).")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for data loading.")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to use for attribution computation.")
    parser.add_argument("--streaming", action="store_true",
                        help="Load the dataset in streaming mode.")
    parser.add_argument("--sparsities", type=str, default="0.5",
                        help="Comma separated list of sparsity fractions for pruning.")
    parser.add_argument("--save_pruned_model", action="store_true",
                        help="Whether to save the pruned model.")
    parser.add_argument("--output_dir", type=str, default="./pruned_models",
                        help="Directory to save the pruned model and evaluation results.")
    parser.add_argument("--test_dataset_name", type=str, default="codeparrot/github-code",
                        help="Dataset name for evaluation.")
    parser.add_argument("--test_split", type=str, default="train",
                        help="Dataset split to use for evaluation.")
    parser.add_argument("--num_test_samples", type=int, default=200,
                        help="Number of samples to use for evaluation.")
    parser.add_argument("--eval_skip", type=int, default=0,
                        help="For streaming datasets, number of samples to skip for evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    assert args.num_samples % args.batch_size == 0, "num_samples must be divisible by batch_size"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map=str(device)
    )

    # Prepare data for attribution score computation.
    print("Preparing attribution data...")
    attribution_dataloader = prepare_data(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        split="train",
        streaming=args.streaming,
        skip_samples=0,
    )
    num_attribution_batches = args.num_samples // args.batch_size
    print("Computing attribution scores...")
    scores = compute_attribution_scores(model, attribution_dataloader, num_attribution_batches)

    # Parse sparsity values (comma separated) into a list of floats.
    sparsity_values = [float(x.strip()) for x in args.sparsities.split(",")]

    # Prepare evaluation data.
    print("Preparing evaluation data...")
    test_dataloader = prepare_data(
        dataset_name=args.test_dataset_name,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_samples=args.num_test_samples,
        split=args.test_split,
        streaming=args.streaming,
        skip_samples=args.eval_skip,
    )

    eval_results = {}
    for sparsity in sparsity_values:
        result_file = os.path.join(args.output_dir, f"sparsity_{sparsity}.json")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Skip if results file already exists
        if os.path.exists(result_file):
            print(f"\nSkipping sparsity {sparsity} as results file already exists: {result_file}")
            continue
            
        pruned_model = copy.deepcopy(model)
        print(f"\nPruning model with sparsity: {sparsity}")
        pruning_stats = prune_model_by_scores(
            model=pruned_model,
            scores=scores,
            sparsity=sparsity,
            save_pruned_model=args.save_pruned_model,
            output_dir=os.path.join(args.output_dir, f"sparsity_{sparsity}")
        )
        print("Evaluating pruned model...")
        eval_stats = evaluate_model(pruned_model, test_dataloader, device)
        eval_stats["pruning_stats"] = {
            "total_neurons_pruned": pruning_stats[0],
            "total_neurons": pruning_stats[1],
            "neurons_pruned_per_layer": pruning_stats[2],
        }
        eval_results[sparsity] = eval_stats
        with open(result_file, "w") as f:
            json.dump(eval_stats, f, indent=4)
        print(f"Evaluation results for sparsity {sparsity} saved to {result_file}")


if __name__ == "__main__":
    main()
