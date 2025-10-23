#!/usr/bin/env python3
"""Analyze learning rate grid search results."""

import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def parse_lr_from_dirname(dirname):
    """Extract learning rate from directory name like 'base_model_lr_1e_5'."""
    match = re.search(r'lr_(\d+)e_?(\d+)', dirname)
    if match:
        mantissa = int(match.group(1))
        exponent = int(match.group(2))
        return mantissa * 10**(-exponent)
    match = re.search(r'lr_(\d+)_(\d+)e(\d+)', dirname)
    if match:
        mantissa = float(f"{match.group(1)}.{match.group(2)}")
        exponent = int(match.group(3))
        return mantissa * 10**(-exponent)
    return None

def load_grid_results(results_dir):
    """Load grid search results organized by model and dataset."""
    results_path = Path(results_dir)
    
    # Group by base model name (without lr suffix)
    models = {}
    
    for model_dir in results_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        dirname = model_dir.name
        
        # Extract base model name and learning rate
        if '_lr_' in dirname:
            base_name = dirname.split('_lr_')[0]
            lr = parse_lr_from_dirname(dirname)
            
            if lr is None:
                continue
            
            if base_name not in models:
                models[base_name] = {}
            
            # Load dataset results
            for dataset_dir in model_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                
                dataset_name = dataset_dir.name
                results_file = dataset_dir / "results.json"
                
                if results_file.exists():
                    with open(results_file) as f:
                        data = json.load(f)
                    
                    if dataset_name not in models[base_name]:
                        models[base_name][dataset_name] = []
                    
                    models[base_name][dataset_name].append({
                        'lr': lr,
                        'baseline_acc': data.get('baseline_accuracy', 0),
                        'final_acc': data.get('final_accuracy', 0),
                        'delta': data.get('final_accuracy', 0) - data.get('baseline_accuracy', 0)
                    })
    
    # Sort by learning rate
    for model_name in models:
        for dataset_name in models[model_name]:
            models[model_name][dataset_name].sort(key=lambda x: x['lr'])
    
    return models

def plot_lr_curves(models, output_dir=None):
    """Plot accuracy vs learning rate curves."""
    datasets = ['counterfact', 'ai2_arc']
    
    for dataset in datasets:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Final accuracy vs LR
        ax1 = axes[0]
        # Plot 2: Accuracy gain vs LR
        ax2 = axes[1]
        
        for model_name in sorted(models.keys()):
            if dataset not in models[model_name]:
                continue
            
            results = models[model_name][dataset]
            lrs = [r['lr'] for r in results]
            final_accs = [r['final_acc'] for r in results]
            deltas = [r['delta'] for r in results]
            
            label = model_name.replace('_model', '').replace('_', ' ')
            
            ax1.plot(lrs, final_accs, marker='o', label=label, linewidth=2)
            ax2.plot(lrs, deltas, marker='o', label=label, linewidth=2)
        
        ax1.set_xscale('log')
        ax1.set_xlabel('Learning Rate')
        ax1.set_ylabel('Final Accuracy')
        ax1.set_title(f'{dataset.upper()} - Final Accuracy vs LR')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xscale('log')
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('Accuracy Gain')
        ax2.set_title(f'{dataset.upper()} - Accuracy Gain vs LR')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir) / f'lr_grid_{dataset}.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {output_path}")
        
        plt.show()

def print_best_lrs(models):
    """Print best learning rates for each model/dataset combination."""
    print("\n" + "="*80)
    print("BEST LEARNING RATES")
    print("="*80)
    
    for model_name in sorted(models.keys()):
        print(f"\n{model_name.upper().replace('_', ' ')}")
        print("-" * 80)
        
        for dataset_name in sorted(models[model_name].keys()):
            results = models[model_name][dataset_name]
            
            # Find best by final accuracy
            best_final = max(results, key=lambda x: x['final_acc'])
            # Find best by accuracy gain
            best_delta = max(results, key=lambda x: x['delta'])
            
            print(f"\n  {dataset_name}:")
            print(f"    Best Final Acc:  lr={best_final['lr']:.2e} -> acc={best_final['final_acc']:.4f}")
            print(f"    Best Delta:      lr={best_delta['lr']:.2e} -> Δ={best_delta['delta']:+.4f}")
            
            # Show all results
            print(f"    All results:")
            for r in results:
                print(f"      lr={r['lr']:.2e}: baseline={r['baseline_acc']:.4f}, "
                      f"final={r['final_acc']:.4f}, Δ={r['delta']:+.4f}")
    
    print("\n" + "="*80 + "\n")

def main():
    import sys
    
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("Loading grid search results...")
    models = load_grid_results(results_dir)
    
    if not models:
        print("No grid search results found!")
        print("Make sure directory names contain '_lr_' suffix")
        return
    
    print(f"Found {len(models)} models with grid search results")
    
    print_best_lrs(models)
    plot_lr_curves(models, output_dir)

if __name__ == "__main__":
    main()



