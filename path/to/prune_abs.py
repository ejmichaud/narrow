#!/usr/bin/env python3
"""
Prune a transformer model using one of two strategies (“attribution” or “weight_norm”)
and evaluate the pruned model on a test dataset. Evaluation statistics are saved
to disk.

python3 prune_abs.py --model_name /afs/csail.mit.edu/u/a/asher/narrow/checkpoint-5000 \
  --pruning_strategy attribution \
  --max_length 512 \
  --batch_size 4 \
  --num_samples 2048 \
  --streaming \
  --sparsity 0.3 \
  --sparsities 0.1 0.3 0.5 \
  --num_test_samples 2048 \
  --eval_skip 2048
"""

import argparse
import json
import os
import copy
from collections.abc import Mapping
from collections import defaultdict
from typing import Any, Dict, Tuple, Union

from tqdm.auto import tqdm
import numpy as np
import einops
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


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


# --- New helper functions for attribution pruning with multiple sparsities ---

def get_attribution_hook(cache: dict, name: Union[int, str], hook_cache: dict):
    """
    A hook to compute attributions. This is registered at each layer.
    """
    def attribution_hook(module, input, output):
        def backward_hook(grad):
            modified_grad = -output.detach() * grad
            cache[name] = modified_grad
            return grad

        hook = output.register_hook(backward_hook)
        hook_cache[name] = hook
        return None

    return attribution_hook


def compute_attribution_scores(
    model: nn.Module, train_dataloader: DataLoader, num_batches: int
) -> list:
    """
    Computes attribution scores across several batches (using hooks) and returns a sorted
    list of tuples (layer_idx, neuron_idx, score).

    The lower the score, the more likely the neuron is pruneable.

    Args:
         model: The language model.
         train_dataloader: DataLoader providing batches from the pruning data.
         num_batches: Number of batches to iterate.

    Returns:
         A sorted list of (layer_idx, neuron_idx, score) tuples.
    """
    scores = {layeri: 0 for layeri in range(len(model.model.layers))}
    total_activations = {layeri: 0 for layeri in range(len(model.model.layers))}
    device = next(model.parameters()).device

    for i, batch in enumerate(tqdm(train_dataloader, desc="computing attribution scores...")):
        if i >= num_batches:
            break

        cache = {}
        forward_hooks = {}
        backward_handles = {}
        for layeri in range(len(model.model.layers)):
            forward_hooks[layeri] = model.model.layers[layeri].mlp.act_fn.register_forward_hook(
                get_attribution_hook(cache, layeri, backward_handles)
            )

        batch = move_to_device(batch, device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        for layeri in range(len(model.model.layers)):
            attrs = cache[layeri]
            scores[layeri] += attrs.sum(dim=tuple(range(attrs.ndim - 1))).detach().abs()
            total_activations[layeri] += attrs.shape[0] * attrs.shape[1]
            forward_hooks[layeri].remove()
            backward_handles[layeri].remove()

        del cache, forward_hooks, backward_handles

    for layeri in scores:
        scores[layeri] /= total_activations[layeri]

    score_tuples = []
    for layeri in range(len(model.model.layers)):
        for neuron_idx in range(scores[layeri].shape[0]):
            score_tuples.append((layeri, neuron_idx, scores[layeri][neuron_idx].item()))
    score_tuples.sort(key=lambda x: x[2])
    return score_tuples


def apply_attribution_pruning(
    model: nn.Module, scores: list, sparsity: float
) -> Tuple[int, int, dict]:
    """
    Given a model and sorted attribution scores, prune a fraction (sparsity) of neurons.
    This function modifies the model *in-place* and returns statistics.
    
    Args:
         model: The model to prune (ideally a deepcopy of the original).
         scores: A sorted list of (layer_idx, neuron_idx, score) tuples.
         sparsity: Fraction of neurons to prune.
    Returns:
         A tuple (total_neurons_pruned, total_neurons, neurons_pruned_per_layer).
    """
    total_neurons = len(scores)
    num_pruned = int(sparsity * total_neurons)
    neurons_pruned_per_layer = defaultdict(int)
    total_neurons_pruned = 0

    for i in range(num_pruned):
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

    return total_neurons_pruned, total_neurons, neurons_pruned_per_layer


# --- Original functions remain for single-sparsity cases ---

def prune_by_attribution(
    model: nn.Module,
    train_dataloader: DataLoader,
    sparsity: float,
    num_attribution_batches: int,
    save_pruned_model: bool,
    output_dir: str,
):
    """Prune neurons based on their attribution scores."""
    # (Original function contents.)
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
            forward_hooks[layeri] = model.model.layers[
                layeri
            ].mlp.act_fn.register_forward_hook(
                get_attribution_hook(cache, layeri, backward_handles)
            )

        batch = move_to_device(batch, next(model.parameters()).device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # remove hooks and sum attribution scores across batch and seq dim
        for layeri in range(len(model.model.layers)):
            attrs = cache[layeri]
            scores[layeri] += attrs.sum(dim=tuple(range(attrs.ndim - 1))).detach().abs()
            total_activations[layeri] += attrs.shape[0] * attrs.shape[1]
            forward_hooks[layeri].remove()
            backward_handles[layeri].remove()

        del cache, forward_hooks, backward_handles

    # average scores
    for layeri in scores:
        scores[layeri] /= total_activations[layeri]

    score_tuples = []
    for layeri in range(len(model.model.layers)):
        for i in range(scores[layeri].shape[0]):
            score_tuples.append((layeri, i, scores[layeri][i].item()))
    scores = score_tuples

    # sort the scores and prune the lowest sparsity fraction
    scores.sort(key=lambda x: x[2])
    num_pruned = int(sparsity * len(scores))
    neurons_pruned_per_layer = defaultdict(int)
    total_neurons_pruned = 0
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

    if save_pruned_model:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        print(f"Pruned model saved to {output_dir}")

    return total_neurons_pruned, total_neurons, neurons_pruned_per_layer


def prune_by_weight_norm(
    model: nn.Module,
    sparsity: float,
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
        # default="NousResearch/Llama-3.2-1B",
        default="/afs/csail.mit.edu/u/a/asher/narrow/experiments/weightpruning1/logs/checkpoint-2000",
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
    # Single sparsity for backward compatibility:
    parser.add_argument(
        "--sparsity", type=float, default=0.5, help="Fraction of neurons to prune."
    )
    # New argument: list of sparsity levels.
    parser.add_argument(
        "--sparsities",
        type=float,
        nargs="+",
        default=None,
        help="List of sparsity levels (fractions between 0 and 1) to evaluate. "
             "If provided, the computed attribution scores will be reused for each sparsity level."
    )
    parser.add_argument(
        "--save_pruned_model",
        action="store_true",
        help="Whether to save the pruned model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./pruned_models",
        help="Directory to save the pruned model and evaluation results.",
    )

    # Evaluation parameters
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
    # New argument: how many samples to skip for evaluation (if using the same split).
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

    # If using attribution for pruning.
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
            skip_samples=0,  # For attribution, start at the beginning.
        )
        num_attribution_batches = args.num_samples // args.batch_size

        # If a list of sparsities is provided, do multi-sparsity evaluation.
        if args.sparsities is not None:
            print("Computing attribution scores...")
            scores = compute_attribution_scores(model, pruning_dataloader, num_attribution_batches)
            total_neurons = len(scores)

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

            multi_sparsity_results = {}
            for s in args.sparsities:
                # Make a deepcopy so that each sparsity evaluation starts from the original model.
                pruned_model = copy.deepcopy(model)
                neurons_pruned, tot_neurons, pruned_per_layer = apply_attribution_pruning(pruned_model, scores, s)
                print(
                    f"Sparsity {s}: pruned {neurons_pruned} / {tot_neurons} "
                    f"({neurons_pruned/tot_neurons:.2%})"
                )
                print("Evaluating pruned model...")
                eval_stats = evaluate_model(pruned_model, test_dataloader, device)
                eval_stats["pruning_stats"] = {
                    "sparsity": s,
                    "total_neurons_pruned": neurons_pruned,
                    "total_neurons": tot_neurons,
                    "neurons_pruned_per_layer": pruned_per_layer,
                }
                multi_sparsity_results[str(s)] = eval_stats
                if args.save_pruned_model:
                    model_save_dir = os.path.join(args.output_dir, f"pruned_model_sparsity_{s}")
                    os.makedirs(model_save_dir, exist_ok=True)
                    pruned_model.save_pretrained(model_save_dir)
                    print(f"Pruned model for sparsity {s} saved to {model_save_dir}")

            # Save evaluation statistics for all sparsities.
            os.makedirs(args.output_dir, exist_ok=True)
            eval_file = os.path.join(args.output_dir, "evaluation_results.json")
            with open(eval_file, "w") as f:
                json.dump(multi_sparsity_results, f, indent=4)
            print(f"Evaluation statistics saved to {eval_file}")
            return  # Exit after multi-sparsity evaluation

        else:
            # Single sparsity case.
            print("Preparing pruning data...")
            pruning_stats = prune_by_attribution(
                model=model,
                train_dataloader=pruning_dataloader,
                sparsity=args.sparsity,
                num_attribution_batches=num_attribution_batches,
                save_pruned_model=args.save_pruned_model,
                output_dir=args.output_dir,
            )

    elif args.pruning_strategy == "weight_norm":
        pruning_stats = prune_by_weight_norm(
            model=model,
            sparsity=args.sparsity,
            save_pruned_model=args.save_pruned_model,
            output_dir=args.output_dir,
        )
    else:
        print("Invalid pruning strategy selected.")
        return

    # Load evaluation data.
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