#!/usr/bin/env python3
"""
Create randomly pruned versions of the base model at different sparsity levels.
This script prunes neurons randomly (not based on importance) and saves the resulting models.
"""

import os
import argparse
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def randomly_prune_neurons(model, neuron_sparsity, seed=42):
    """
    Randomly prune neurons from the model.
    
    Args:
        model: The model to prune
        neuron_sparsity: Fraction of neurons to prune (0.0 to 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        pruned_neurons: List of (layer_idx, neuron_idx) tuples that were pruned
    """
    np.random.seed(seed)
    
    # Count total neurons
    n_neurons = sum(layer.mlp.gate_proj.out_features for layer in model.model.layers)
    neurons_to_prune_count = int(n_neurons * neuron_sparsity)
    
    print(f"\n=== Random Pruning ===")
    print(f"Total neurons: {n_neurons}")
    print(f"Target sparsity: {neuron_sparsity:.1%}")
    print(f"Neurons to prune: {neurons_to_prune_count}")
    
    # Randomly select neurons to prune
    pruned_neurons = []
    pruned_set = set()
    
    while len(pruned_neurons) < neurons_to_prune_count:
        layeri = np.random.randint(0, len(model.model.layers))
        neuroni = np.random.randint(0, model.config.intermediate_size)
        
        # Avoid duplicates
        if (layeri, neuroni) not in pruned_set:
            pruned_neurons.append((layeri, neuroni))
            pruned_set.add((layeri, neuroni))
    
    # Apply pruning by zeroing out weights
    with torch.no_grad():
        for layeri, neuroni in pruned_neurons:
            model.model.layers[layeri].mlp.gate_proj.weight[neuroni, :] = 0
            model.model.layers[layeri].mlp.up_proj.weight[neuroni, :] = 0
            model.model.layers[layeri].mlp.down_proj.weight[:, neuroni] = 0
    
    print(f"Successfully pruned {len(pruned_neurons)} neurons")
    
    # Count neurons pruned per layer
    neurons_per_layer = {}
    for layeri, neuroni in pruned_neurons:
        neurons_per_layer[layeri] = neurons_per_layer.get(layeri, 0) + 1
    
    print(f"Neurons pruned per layer (showing non-zero only):")
    for layeri in sorted(neurons_per_layer.keys()):
        print(f"  Layer {layeri}: {neurons_per_layer[layeri]} / {model.config.intermediate_size}")
    
    return pruned_neurons, neurons_per_layer


def main():
    parser = argparse.ArgumentParser(description="Create randomly pruned models")
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
        "--output_base_dir",
        type=str,
        default=None,
        help="Base output directory (defaults to $SCRATCH/iaifi_lab/Lab/ericjm/narrow/random_pruned)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set up output directory
    if args.output_base_dir is None:
        scratch = os.environ.get('SCRATCH', '/tmp')
        args.output_base_dir = os.path.join(scratch, 'iaifi_lab/Lab/ericjm/narrow/random_pruned')
    
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    print(f"{'='*80}")
    print(f"Random Neuron Pruning")
    print(f"{'='*80}")
    print(f"Base model: {args.model_name}")
    print(f"Sparsity levels: {args.sparsity_levels}")
    print(f"Output directory: {args.output_base_dir}")
    print(f"Random seed: {args.seed}")
    print(f"{'='*80}\n")
    
    # Load tokenizer once
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Process each sparsity level
    for sparsity in args.sparsity_levels:
        print(f"\n{'='*80}")
        print(f"Processing sparsity level: {sparsity:.2%}")
        print(f"{'='*80}")
        
        # Load fresh model for each sparsity level
        print(f"Loading model: {args.model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            device_map="cpu"  # Keep on CPU to save GPU memory
        )
        
        # Randomly prune neurons
        pruned_neurons, neurons_per_layer = randomly_prune_neurons(
            model, 
            neuron_sparsity=sparsity,
            seed=args.seed
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
            "neuron_sparsity": sparsity,
            "total_neurons": sum(layer.mlp.gate_proj.out_features for layer in model.model.layers),
            "neurons_pruned": len(pruned_neurons),
            "neurons_per_layer": neurons_per_layer,
            "random_seed": args.seed,
            "pruned_neurons": pruned_neurons  # List of (layer, neuron) tuples
        }
        
        stats_file = os.path.join(output_dir, "pruning_stats.json")
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved pruning statistics to: {stats_file}")
        
        # Clean up to free memory
        del model
        torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print(f"All models created successfully!")
    print(f"{'='*80}")
    print(f"\nOutput directory structure:")
    for sparsity in args.sparsity_levels:
        output_dir = os.path.join(args.output_base_dir, f"sparsity_{sparsity}")
        print(f"  {output_dir}/")


if __name__ == "__main__":
    main()


