#!/usr/bin/env python3
"""
Prune a transformer model using one of two strategies (“attribution” or “weight_norm”) and evaluate the pruned model on a test dataset. Evaluation statistics are saved to disk.

Example (attribution pruning):

python3 prune_abs.py --model_name NousResearch/Llama-3.2-1B \
    --pruning_strategy attribution \
    --max_length 512 \
    --batch_size 8 \
    --num_samples 2048 \
    --streaming \
    --sparsities 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
    --num_test_samples 2048 \
    --eval_skip 2048
    --output_dir ./pruned_models/base
"""

import argparse
import json
import os
import copy
from collections.abc import Mapping
from collections import defaultdict
from typing import Any, Dict, Tuple

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"


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
        dataset = load_dataset(
            dataset_name, split=split, languages=["Python"], streaming=True
        )
        if skip_samples > 0:
            dataset = dataset.skip(skip_samples)
        dataset = dataset.take(num_samples)
        # Convert streaming dataset to a list to allow tokenization.
        dataset = list(dataset)
    else:
        dataset = load_dataset(dataset_name, split=split, languages=["Python"])
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
        else dataset.map(
            tokenize_function, batched=True, remove_columns=dataset.column_names
        )
    )
    # If tokenized_dataset is a list (from streaming), tokenize each sample.
    if isinstance(tokenized_dataset, list):
        tokenized_dataset = [tokenize_function(sample) for sample in tokenized_dataset]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        tokenized_dataset, batch_size=batch_size, collate_fn=data_collator
    )
    return dataloader


@torch.no_grad()
def evaluate_model(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> Dict[str, float]:
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


def prune_by_weight_norm(
    model: nn.Module,
    sparsity: float,
    save_pruned_model: bool,
    output_dir: str,
) -> Tuple[int, int, list]:
    """
    Prune neurons by comparing the L2 norm of their associated weights to a threshold.

    Returns:
        A tuple (total_neurons_pruned, total_neurons, neurons_pruned_per_layer) across all layers.
    """
    total_neurons_pruned = 0
    total_neurons = 0
    neurons_pruned_per_layer = []
    scores = []

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
        score_tuples = [(layer_idx, i, L2[i].item()) for i in range(L2.shape[0])]
        scores.extend(score_tuples)

    # sort the scores and prune the lowest sparsity fraction
    neurons_pruned_per_layer = defaultdict(int)
    scores.sort(key=lambda x: x[2])
    num_pruned = int(sparsity * len(scores))
    for i in tqdm(range(num_pruned), desc="pruning neurons..."):
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

    total_neurons = len(scores)

    print(
        f"Total neurons pruned: {total_neurons_pruned} / {total_neurons} "
        f"({total_neurons_pruned/total_neurons:.2%})"
    )
    neurons_pruned_per_layer = [
        neurons_pruned_per_layer[i] for i in range(len(model.model.layers))
    ]

    if save_pruned_model:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        print(f"Pruned model saved to {output_dir}")

    return total_neurons_pruned, total_neurons, neurons_pruned_per_layer


def compute_attribution_scores(
    model: nn.Module,
    train_dataloader: DataLoader,
    num_attribution_batches: int,
    device: torch.device,
):
    """
    Compute attribution scores for each neuron in the model's MLP layers.

    Returns:
         sorted_scores: List of tuples (layer_idx, neuron_idx, score) sorted by score ascending.
         total_neurons: Total number of neurons considered.
    """
    model.train()

    def get_attribution_hook(cache, name, hook_cache):
        def attribution_hook(module, input, output):
            def backward_hook(grad):
                modified_grad = -output.detach() * grad
                cache[name] = modified_grad
                return grad

            hook_cache[name] = output.register_hook(backward_hook)
            return None

        return attribution_hook

    num_layers = len(model.model.layers)
    scores = {layeri: None for layeri in range(num_layers)}
    total_activations = {layeri: 0 for layeri in range(num_layers)}

    for i, batch in enumerate(
        tqdm(train_dataloader, desc="Computing attribution scores...")
    ):
        if i >= num_attribution_batches:
            break

        cache = {}
        forward_hooks = {}
        backward_handles = {}
        for layeri in range(num_layers):
            forward_hooks[layeri] = model.model.layers[
                layeri
            ].mlp.act_fn.register_forward_hook(
                get_attribution_hook(cache, layeri, backward_handles)
            )

        batch = move_to_device(batch, device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        for layeri in range(num_layers):
            if layeri in cache:
                attr = cache[layeri]
                # Sum attribution scores across batch and sequence dimensions.
                summed_attr = attr.sum(dim=tuple(range(attr.ndim - 1))).detach().abs()
                if scores[layeri] is None:
                    scores[layeri] = summed_attr
                else:
                    scores[layeri] += summed_attr
                total_activations[layeri] += attr.shape[0] * attr.shape[1]
            forward_hooks[layeri].remove()
            if layeri in backward_handles:
                backward_handles[layeri].remove()

        del cache, forward_hooks, backward_handles

    sorted_scores = []
    for layeri in range(num_layers):
        scores[layeri] /= total_activations[layeri]
        for neuron_idx in range(scores[layeri].shape[0]):
            sorted_scores.append(
                (layeri, neuron_idx, scores[layeri][neuron_idx].item())
            )
    sorted_scores.sort(key=lambda x: x[2])
    total_neurons = len(sorted_scores)
    return sorted_scores, total_neurons


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prune a language model using attribution or weight norm and evaluate it."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="NousResearch/Llama-3.2-1B",
        help="Pretrained model name or path.",
    )
    parser.add_argument(
        "--pruning_strategy",
        type=str,
        choices=["attribution", "weight_norm"],
        default="attribution",
        help="Pruning strategy to use.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="codeparrot/github-code",
        help="Dataset name for pruning (training data).",
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for data loading."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to use for pruning data.",
    )
    parser.add_argument(
        "--streaming", action="store_true", help="Load the dataset in streaming mode."
    )
    # Provide a list of sparsity levels.
    parser.add_argument(
        "--sparsities",
        type=float,
        nargs="+",
        default=[0.5],
        help="List of fractions of neurons to prune. For attribution, neurons are removed sequentially.",
    )
    parser.add_argument(
        "--save_pruned_model",
        action="store_true",
        help="Whether to save the pruned model at each sparsity level.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./pruned_models",
        help="Directory to save the pruned model and evaluation results.",
    )
    parser.add_argument(
        "--test_dataset_name",
        type=str,
        default="codeparrot/github-code",
        help="Dataset name for evaluation.",
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="train",
        help="Dataset split to use for evaluation.",
    )
    parser.add_argument(
        "--num_test_samples",
        type=int,
        default=200,
        help="Number of samples to use for evaluation.",
    )
    parser.add_argument(
        "--eval_skip",
        type=int,
        default=0,
        help="For streaming datasets, number of samples to skip for evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    assert (
        args.num_samples % args.batch_size == 0
    ), "num_samples must be divisible by batch_size"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model.
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, device_map=str(device)
    )

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

    if args.pruning_strategy == "attribution":
        # Prepare pruning data.
        print("Preparing pruning data for attribution...")
        pruning_dataloader = prepare_data(
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
        # Compute attribution scores once.
        sorted_scores, total_neurons = compute_attribution_scores(
            model, pruning_dataloader, num_attribution_batches, device
        )
        print(f"Computed attribution scores over {total_neurons} neurons.")

        # For sequential pruning, sort the sparsity levels in ascending order.
        sparsities = sorted(args.sparsities)
        # Start with a deep copy of the original model.
        pruned_model = copy.deepcopy(model).to(device)
        total_neurons_pruned = 0  # cumulative count of pruned neurons
        eval_results = {}

        for sparsity in sparsities:
            # Determine desired total neurons pruned at this sparsity.
            target_pruned = int(total_neurons * sparsity)
            additional_to_prune = target_pruned - total_neurons_pruned

            if additional_to_prune > 0:
                print(
                    f"\nIncreasing sparsity to {sparsity} "
                    f"({target_pruned} neurons total, pruning additional {additional_to_prune} neurons)..."
                )
                # Sequentially remove the additional neurons based on sorted attribution scores.
                for i in range(total_neurons_pruned, target_pruned):
                    layer_idx, neuron_idx, _ = sorted_scores[i]
                    layer = pruned_model.model.layers[layer_idx]
                    layer.mlp.gate_proj.weight.data[neuron_idx] = 0.0
                    layer.mlp.up_proj.weight.data[neuron_idx] = 0.0
                    layer.mlp.down_proj.weight.data[:, neuron_idx] = 0.0
                total_neurons_pruned = target_pruned
            else:
                print(f"\nSparsity {sparsity} reached with no additional pruning.")

            print(
                f"Total neurons pruned: {total_neurons_pruned} / {total_neurons} "
                f"({total_neurons_pruned/total_neurons:.2%})"
            )
            if args.save_pruned_model:
                out_dir = os.path.join(
                    args.output_dir, f"attribution_sparsity_{int(sparsity*100)}"
                )
                os.makedirs(out_dir, exist_ok=True)
                pruned_model.save_pretrained(out_dir)
                print(f"Sequentially pruned model saved to {out_dir}")

            print(f"Evaluating pruned model at sparsity {sparsity}...")
            eval_stats = evaluate_model(pruned_model, test_dataloader, device)
            eval_stats["pruning_stats"] = {
                "total_neurons_pruned": total_neurons_pruned,
                "total_neurons": total_neurons,
            }
            eval_results[sparsity] = eval_stats

        # Save evaluation results for attribution pruning.
        os.makedirs(args.output_dir, exist_ok=True)
        eval_file = os.path.join(args.output_dir, "evaluation_results_attribution.json")

        # Load existing data if file exists
        existing_data = {}
        if os.path.exists(eval_file):
            with open(eval_file, "r") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    print(
                        f"Warning: Could not parse existing file {eval_file}. Creating new file."
                    )

        # Update with new results
        existing_data.update(eval_results)

        # Write combined data back to file
        with open(eval_file, "w") as f:
            json.dump(existing_data, f, indent=4)
        print(f"Evaluation statistics appended to {eval_file}")

    elif args.pruning_strategy == "weight_norm":
        # If multiple sparsities are given for weight_norm, only the first one is used.
        if len(args.sparsities) > 1:
            print(
                "Multiple sparsities specified, but weight_norm only supports a single sparsity. Using the first one."
            )
        pruning_stats = prune_by_weight_norm(
            model=model,
            sparsity=args.sparsities[0],
            save_pruned_model=args.save_pruned_model,
            output_dir=args.output_dir,
        )

        print("Evaluating pruned model...")
        eval_stats = evaluate_model(model, test_dataloader, device)
        eval_stats["pruning_stats"] = {
            "total_neurons_pruned": pruning_stats[0],
            "total_neurons": pruning_stats[1],
            "neurons_pruned_per_layer": pruning_stats[2],
        }
        os.makedirs(args.output_dir, exist_ok=True)
        eval_file = os.path.join(args.output_dir, "evaluation_results_weight_norm.json")

        # Load existing data if file exists
        existing_data = {}
        if os.path.exists(eval_file):
            with open(eval_file, "r") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    print(
                        f"Warning: Could not parse existing file {eval_file}. Creating new file."
                    )

        # Add current sparsity level result to existing data
        sparsity_key = str(args.sparsities[0])  # Use sparsity as key
        existing_data[sparsity_key] = eval_stats

        # Write combined data back to file
        with open(eval_file, "w") as f:
            json.dump(existing_data, f, indent=4)
        print(f"Evaluation statistics appended to {eval_file}")
    else:
        print("Invalid pruning strategy selected.")
        return


if __name__ == "__main__":
    main()
