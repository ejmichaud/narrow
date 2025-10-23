#!/usr/bin/env python3
"""View and compare results from unlearning experiments."""

import json
import os
from pathlib import Path

def load_results(results_dir):
    """Load all results.json files from results directory."""
    results = {}
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return results
    
    for model_dir in results_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        results[model_name] = {}
        
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset_name = dataset_dir.name
            results_file = dataset_dir / "results.json"
            
            if results_file.exists():
                with open(results_file) as f:
                    results[model_name][dataset_name] = json.load(f)
    
    return results

def print_results(results):
    """Pretty print results in a table."""
    if not results:
        print("No results found.")
        return
    
    print("\n" + "="*80)
    print("UNLEARNING EXPERIMENT RESULTS")
    print("="*80)
    
    for model_name, datasets in sorted(results.items()):
        print(f"\n{model_name.upper().replace('_', ' ')}")
        print("-" * 80)
        
        for dataset_name, data in sorted(datasets.items()):
            baseline = data.get('baseline_accuracy', 0)
            final = data.get('final_accuracy', 0)
            change = final - baseline
            
            print(f"\n  {dataset_name}:")
            print(f"    Baseline:  {baseline:.3f}")
            print(f"    Final:     {final:.3f}")
            print(f"    Change:    {change:+.3f} ({change/baseline*100:+.1f}%)")
            print(f"    Model:     {data.get('model_path', 'N/A')}")
            
            if 'learning_rate' in data:
                print(f"    LR:        {data['learning_rate']}")
            if 'epochs' in data:
                print(f"    Epochs:    {data['epochs']}")
    
    print("\n" + "="*80)
    
    # Summary comparison
    print("\nSUMMARY COMPARISON")
    print("-" * 80)
    
    datasets_found = set()
    for datasets in results.values():
        datasets_found.update(datasets.keys())
    
    # Sort models: base first, then lambda values, then random sparsity
    def sort_key(model_name):
        import re
        if 'base' in model_name:
            return (0, 0.0, '')
        elif 'lambda' in model_name:
            # Extract lambda value for sorting
            match = re.search(r'lambda_([\d.]+)', model_name)
            if match:
                return (1, float(match.group(1)), '')
        elif 'sparsity' in model_name:
            # Extract sparsity value for sorting
            match = re.search(r'sparsity_([\d.]+)', model_name)
            if match:
                return (2, float(match.group(1)), '')
        return (3, 0.0, model_name)
    
    sorted_models = sorted(results.keys(), key=sort_key)
    
    for dataset in sorted(datasets_found):
        print(f"\n{dataset}:")
        print(f"  {'Model':<30} {'Baseline':>8} {'Final':>8} {'Change':>10}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*10}")
        for model_name in sorted_models:
            if dataset in results[model_name]:
                data = results[model_name][dataset]
                baseline = data.get('baseline_accuracy', 0)
                final = data.get('final_accuracy', 0)
                change = final - baseline
                print(f"  {model_name:<30} {baseline:>8.3f} {final:>8.3f} {change:>+10.3f}")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    import sys
    
    # Default to results directory in current directory
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    
    results = load_results(results_dir)
    print_results(results)


