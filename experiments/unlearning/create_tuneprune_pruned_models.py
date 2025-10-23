#!/usr/bin/env python3
"""
Create pruned versions of tuneprune15 models by thresholding small weights.
Maps: lambda_0.0003 → 30%, lambda_0.0005 → 63%, lambda_0.001 → 80% sparsity
"""

import os
import argparse
import json
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def prune_by_magnitude(model, target_sparsity):
    """
    Prune neurons with smallest L2 norm to achieve target sparsity.
    
    Returns:
        pruned_neurons: List of (layer_idx, neuron_idx) tuples
        neurons_per_layer: Dict of neurons pruned per layer
    """
    # Compute L2 norm for each neuron across all three MLP matrices
    neuron_norms = []
    
    for layeri, layer in enumerate(model.model.layers):
        gate_norms = layer.mlp.gate_proj.weight.norm(dim=1)  # [intermediate_size]
        up_norms = layer.mlp.up_proj.weight.norm(dim=1)
        down_norms = layer.mlp.down_proj.weight.norm(dim=0)
        
        # Combined norm for each neuron
        combined_norms = (gate_norms.pow(2) + up_norms.pow(2) + down_norms.pow(2)).sqrt()
        
        for neuroni in range(combined_norms.shape[0]):
            neuron_norms.append((layeri, neuroni, combined_norms[neuroni].item()))
    
    # Sort by norm (smallest first) and prune bottom sparsity%
    neuron_norms.sort(key=lambda x: x[2])
    num_to_prune = int(target_sparsity * len(neuron_norms))
    
    print(f"\nPruning {num_to_prune} / {len(neuron_norms)} neurons ({target_sparsity:.1%})")
    
    pruned_neurons = []
    neurons_per_layer = defaultdict(int)
    
    # Prune neurons with smallest norms
    with torch.no_grad():
        for i in range(num_to_prune):
            layeri, neuroni, norm = neuron_norms[i]
            
            model.model.layers[layeri].mlp.gate_proj.weight[neuroni, :] = 0
            model.model.layers[layeri].mlp.up_proj.weight[neuroni, :] = 0
            model.model.layers[layeri].mlp.down_proj.weight[:, neuroni] = 0
            
            pruned_neurons.append((layeri, neuroni))
            neurons_per_layer[layeri] += 1
    
    print(f"Neurons pruned per layer (showing non-zero only):")
    for layeri in sorted(neurons_per_layer.keys()):
        total = model.config.intermediate_size
        print(f"  Layer {layeri}: {neurons_per_layer[layeri]} / {total}")
    
    return pruned_neurons, dict(neurons_per_layer)


def main():
    parser = argparse.ArgumentParser(description="Create magnitude-pruned tuneprune models")
    parser.add_argument(
        "--tuneprune_dir",
        type=str,
        default=None,
        help="Base directory for tuneprune15-redo models"
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=None,
        help="Base output directory"
    )
    
    args = parser.parse_args()
    
    # Set up directories
    scratch = os.environ.get('SCRATCH', '/tmp')
    if args.tuneprune_dir is None:
        args.tuneprune_dir = os.path.join(scratch, 'iaifi_lab/Lab/ericjm/narrow/tuneprune15-redo')
    if args.output_base_dir is None:
        args.output_base_dir = os.path.join(scratch, 'iaifi_lab/Lab/ericjm/narrow/tuneprune_pruned')
    
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    # Define mapping: lambda → target sparsity
    lambda_sparsity_map = [
        ("lambda_0.0003_bs_18_acc_6", 0.3),
        ("lambda_0.0005_bs_18_acc_6", 0.63),
        ("lambda_0.001_bs_18_acc_6", 0.8),
    ]
    
    print(f"{'='*80}")
    print(f"TunePrune Model Pruning")
    print(f"{'='*80}")
    print(f"TunePrune directory: {args.tuneprune_dir}")
    print(f"Output directory: {args.output_base_dir}")
    print(f"{'='*80}\n")
    
    for lambda_name, target_sparsity in lambda_sparsity_map:
        print(f"\n{'='*80}")
        print(f"Processing: {lambda_name} → {target_sparsity:.1%} sparsity")
        print(f"{'='*80}")
        
        # Load model
        model_path = os.path.join(args.tuneprune_dir, lambda_name, "checkpoint-70000")
        
        if not os.path.exists(model_path):
            print(f"WARNING: Model not found at {model_path}, skipping...")
            continue
        
        print(f"Loading model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Prune by magnitude
        pruned_neurons, neurons_per_layer = prune_by_magnitude(model, target_sparsity)
        
        # Save pruned model
        output_name = f"{lambda_name}_sparsity_{target_sparsity}"
        output_dir = os.path.join(args.output_base_dir, output_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving model to: {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save statistics
        stats = {
            "source_model": model_path,
            "lambda_value": lambda_name,
            "pruning_method": "magnitude",
            "neuron_sparsity": target_sparsity,
            "total_neurons": sum(layer.mlp.gate_proj.out_features for layer in model.model.layers),
            "neurons_pruned": len(pruned_neurons),
            "neurons_per_layer": neurons_per_layer,
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
    print(f"All tuneprune-pruned models created successfully!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()






