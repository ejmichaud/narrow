#!/usr/bin/env python3
"""
Create attribution-based pruned versions of the base model at different sparsity levels.
Prunes neurons based on their importance to Python code generation.
"""

import os
import sys
import argparse
import json
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


def move_to_device(data, device):
    """Recursively move data to device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)([move_to_device(v, device) for v in data])
    return data


def prepare_python_dataloader(model_name, num_samples=1024, batch_size=8, max_length=1024):
    """Load Python code from the-stack for attribution computation."""
    print(f"Loading {num_samples} Python code samples...")
    
    dataset = load_dataset(
        "codeparrot/github-code",
        split="train",
        languages=["Python"],
        streaming=True,
        trust_remote_code=True
    )
    dataset = dataset.take(num_samples)
    dataset = list(dataset)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(
            examples["code"],
            truncation=True,
            max_length=max_length,
        )
    
    tokenized_dataset = [tokenize_function(sample) for sample in dataset]
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)
    
    return dataloader


def compute_attribution_scores(model, dataloader, num_batches):
    """
    Compute attribution scores for neurons based on Python code.
    Attribution = -activation * gradient_of_loss
    """
    print(f"Computing attribution scores on {num_batches} batches...")
    
    def get_attribution_hook(cache, name, hook_cache):
        def attribution_hook(module, input, output):
            def backward_hook(grad):
                # Attribution: -activation * gradient
                modified_grad = -output.detach() * grad
                cache[name] = modified_grad
                return grad
            hook_cache[name] = output.register_hook(backward_hook)
            return None
        return attribution_hook
    
    scores = {layeri: 0 for layeri in range(len(model.model.layers))}
    total_activations = {layeri: 0 for layeri in range(len(model.model.layers))}
    
    # Get device from model
    device = next(model.parameters()).device
    
    for i, batch in enumerate(tqdm(dataloader, desc="Computing attribution")):
        if i >= num_batches:
            break
        
        cache = {}
        forward_hooks = {}
        backward_handles = {}
        
        # Register hooks on MLP activation functions
        for layeri in range(len(model.model.layers)):
            forward_hooks[layeri] = model.model.layers[layeri].mlp.act_fn.register_forward_hook(
                get_attribution_hook(cache, layeri, backward_handles)
            )
        
        # Move batch to device - ensure all dict values are moved
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        # Aggregate attribution scores
        for layeri in range(len(model.model.layers)):
            attrs = cache[layeri]
            scores[layeri] += attrs.sum(dim=tuple(range(attrs.ndim - 1))).detach().abs()
            total_activations[layeri] += attrs.shape[0] * attrs.shape[1]
            forward_hooks[layeri].remove()
            backward_handles[layeri].remove()
        
        del cache
        del forward_hooks
        del backward_handles
        model.zero_grad()
    
    # Average scores
    for layeri in scores:
        scores[layeri] /= total_activations[layeri]
    
    return scores


def prune_by_attribution(model, attribution_scores, sparsity):
    """
    Prune neurons with lowest attribution scores.
    
    Returns:
        pruned_neurons: List of (layer_idx, neuron_idx) tuples
        neurons_per_layer: Dict of neurons pruned per layer
    """
    # Create list of (layer, neuron, score) tuples
    score_tuples = []
    for layeri in range(len(model.model.layers)):
        for neuroni in range(attribution_scores[layeri].shape[0]):
            score_tuples.append((layeri, neuroni, attribution_scores[layeri][neuroni].item()))
    
    # Sort by score (lowest first) and prune bottom sparsity%
    score_tuples.sort(key=lambda x: x[2])
    num_to_prune = int(sparsity * len(score_tuples))
    
    print(f"\nPruning {num_to_prune} / {len(score_tuples)} neurons ({sparsity:.1%})")
    
    pruned_neurons = []
    neurons_per_layer = defaultdict(int)
    
    # Prune lowest-scoring neurons
    with torch.no_grad():
        for i in range(num_to_prune):
            layeri, neuroni, score = score_tuples[i]
            
            model.model.layers[layeri].mlp.gate_proj.weight[neuroni, :] = 0
            model.model.layers[layeri].mlp.up_proj.weight[neuroni, :] = 0
            model.model.layers[layeri].mlp.down_proj.weight[:, neuroni] = 0
            
            pruned_neurons.append((layeri, neuroni))
            neurons_per_layer[layeri] += 1
    
    print(f"Neurons pruned per layer (showing non-zero only):")
    for layeri in sorted(neurons_per_layer.keys()):
        print(f"  Layer {layeri}: {neurons_per_layer[layeri]} / {model.config.intermediate_size}")
    
    return pruned_neurons, dict(neurons_per_layer)


def main():
    parser = argparse.ArgumentParser(description="Create attribution-based pruned models")
    parser.add_argument(
        "--model_name",
        type=str,
        default="NousResearch/Llama-3.2-1B",
        help="Base model to prune"
    )
    parser.add_argument(
        "--sparsity_levels",
        type=float,
        nargs="+",
        default=[0.3, 0.63, 0.8],
        help="Neuron sparsity levels to test"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1024,
        help="Number of Python code samples for attribution"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for attribution computation"
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=None,
        help="Base output directory (defaults to $SCRATCH/iaifi_lab/Lab/ericjm/narrow/attribution_pruned)"
    )
    
    args = parser.parse_args()
    
    # Set up output directory
    if args.output_base_dir is None:
        scratch = os.environ.get('SCRATCH', '/tmp')
        args.output_base_dir = os.path.join(scratch, 'iaifi_lab/Lab/ericjm/narrow/attribution_pruned')
    
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    print(f"{'='*80}")
    print(f"Attribution-Based Neuron Pruning")
    print(f"{'='*80}")
    print(f"Base model: {args.model_name}")
    print(f"Sparsity levels: {args.sparsity_levels}")
    print(f"Attribution samples: {args.num_samples}")
    print(f"Output directory: {args.output_base_dir}")
    print(f"{'='*80}\n")
    
    # Load model once for attribution computation
    print("Loading model for attribution...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    # Prepare Python code data
    dataloader = prepare_python_dataloader(
        args.model_name,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_length=1024
    )
    
    # Compute attribution scores (do this once, use for all sparsity levels)
    num_batches = args.num_samples // args.batch_size
    attribution_scores = compute_attribution_scores(model, dataloader, num_batches)
    
    # Save attribution scores for reference
    attribution_file = os.path.join(args.output_base_dir, "attribution_scores.pt")
    torch.save(attribution_scores, attribution_file)
    print(f"\nSaved attribution scores to: {attribution_file}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Clean up and reload for each sparsity level
    del model
    torch.cuda.empty_cache()
    
    # Process each sparsity level
    for sparsity in args.sparsity_levels:
        print(f"\n{'='*80}")
        print(f"Processing sparsity level: {sparsity:.2%}")
        print(f"{'='*80}")
        
        # Load fresh model
        print(f"Loading fresh model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            device_map="cpu"  # Keep on CPU to save GPU memory
        )
        
        # Prune based on attribution scores
        pruned_neurons, neurons_per_layer = prune_by_attribution(
            model,
            attribution_scores,
            sparsity
        )
        
        # Save pruned model
        output_dir = os.path.join(args.output_base_dir, f"sparsity_{sparsity}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving model to: {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save pruning statistics
        stats = {
            "base_model": args.model_name,
            "pruning_method": "attribution",
            "neuron_sparsity": sparsity,
            "total_neurons": sum(layer.mlp.gate_proj.out_features for layer in model.model.layers),
            "neurons_pruned": len(pruned_neurons),
            "neurons_per_layer": neurons_per_layer,
            "num_attribution_samples": args.num_samples,
            "pruned_neurons": pruned_neurons
        }
        
        stats_file = os.path.join(output_dir, "pruning_stats.json")
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved pruning statistics to: {stats_file}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print(f"All attribution-based models created successfully!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()


