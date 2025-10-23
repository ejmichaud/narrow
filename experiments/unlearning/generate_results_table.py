#!/usr/bin/env python3
"""
Generate NeurIPS-style LaTeX table of pruned model evaluation results.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def load_results(results_dir):
    """Load all results from pruned model evaluations."""
    results = defaultdict(lambda: defaultdict(dict))
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return results
    
    for model_dir in results_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Parse model type and sparsity
        if 'base_model' in model_name:
            method = 'Base'
            sparsity = '0%'
        elif 'random' in model_name:
            method = 'Random'
        elif 'attribution' in model_name:
            method = 'Attribution'
        elif 'tuneprune' in model_name:
            method = 'Group Lasso'
        else:
            continue
        
        # Extract sparsity for pruned models
        if method != 'Base':
            if 'sparsity_0.3' in model_name or '_0.3' in model_name:
                sparsity = '30%'
            elif 'sparsity_0.63' in model_name or '_0.63' in model_name:
                sparsity = '63%'
            elif 'sparsity_0.8' in model_name or '_0.8' in model_name:
                sparsity = '80%'
            else:
                continue
        
        # Load results for each dataset
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset_name = dataset_dir.name
            results_file = dataset_dir / "results.json"
            
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                    key = (method, sparsity)
                    # Use best_accuracy if available (from LR sweep), otherwise final_accuracy
                    final_acc = data.get('best_accuracy', data.get('final_accuracy', 0))
                    results[key][dataset_name] = {
                        'baseline': data.get('baseline_accuracy', 0),
                        'final': final_acc
                    }
    
    return results


def generate_latex_table(results):
    """Generate LaTeX table code."""
    
    # Dataset order and display names
    datasets = [
        ('counterfact', 'CounterFact'),
        ('ai2_arc', 'AI2-ARC'),
        ('wmdp_bio', 'WMDP-Bio'),
        ('wmdp_cyber', 'WMDP-Cyber'),
        ('wmdp_chem', 'WMDP-Chem'),
    ]
    
    methods = ['Base', 'Random', 'Attribution', 'Group Lasso']
    sparsities_map = {
        'Base': ['0%'],
        'Random': ['30%', '63%', '80%'],
        'Attribution': ['30%', '63%', '80%'],
        'Group Lasso': ['30%', '63%', '80%']
    }
    
    # Generate table
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Downstream task performance of pruned models before and after fine-tuning.}")
    lines.append(r"\label{tab:pruned_results}")
    
    # Column specification
    ncols = 2 + len(datasets) * 2  # Method + Sparsity + 2 columns per dataset
    # Use @{} to trim outer padding for more compact layout
    lines.append(r"\begin{tabular}{@{}ll" + "rr" * len(datasets) + "@{}}")
    lines.append(r"\toprule")
    
    # Header
    header = r"\textbf{Method} & \textbf{Sparsity}"
    for _, display_name in datasets:
        header += f" & \multicolumn{{2}}{{c}}{{{display_name}}}"
    header += r" \\"
    lines.append(header)
    
    # Column group rules for dataset pairs
    cmidrules = []
    for i in range(len(datasets)):
        start = 3 + i * 2
        end = start + 1
        # Single backslash in LaTeX output
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
    lines.append(" ".join(cmidrules))

    # Subheader
    subheader = r"& & " + " & ".join(["Base & FT"] * len(datasets)) + r" \\" 
    lines.append(subheader)
    lines.append(r"\midrule")
    
    # Data rows
    for method in methods:
        method_sparsities = sparsities_map[method]
        
        for i, sparsity in enumerate(method_sparsities):
            key = (method, sparsity)
            
            if i == 0:
                row = f"{method}"
            else:
                row = ""
            
            row += f" & {sparsity}"
            
            for dataset_key, _ in datasets:
                if key in results and dataset_key in results[key]:
                    baseline = results[key][dataset_key]['baseline']
                    final = results[key][dataset_key]['final']
                    row += f" & {baseline:.3f} & {final:.3f}"
                else:
                    row += r" & \multicolumn{1}{c}{--} & \multicolumn{1}{c}{--}"
            
            row += r" \\"
            lines.append(row)
        
        if method != methods[-1]:
            lines.append(r"\midrule")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def generate_latex_table_simple(results):
    """Generate a simplified LaTeX table with no multicolumns/cmidrules."""
    # Dataset order and display names (short labels)
    datasets = [
        ('counterfact', 'CF'),
        ('ai2_arc', 'ARC'),
        ('wmdp_bio', 'Bio'),
        ('wmdp_cyber', 'Cyber'),
        ('wmdp_chem', 'Chem'),
    ]

    methods = ['Base', 'Random', 'Attribution', 'Group Lasso']
    sparsities_map = {
        'Base': ['0%'],
        'Random': ['30%', '63%', '80%'],
        'Attribution': ['30%', '63%', '80%'],
        'Group Lasso': ['30%', '63%', '80%']
    }

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Downstream task performance of pruned models before and after fine-tuning.}")
    lines.append(r"\label{tab:pruned_results}")
    # Simple column spec: no @{}, no multicolumns
    lines.append(r"\begin{tabular}{ll" + "rr" * len(datasets) + "}")
    lines.append(r"\toprule")

    # Single-line header with short labels
    header_parts = [r"\textbf{Method}", r"\textbf{Sparsity}"]
    for _, short in datasets:
        header_parts.append(fr"{short} Base")
        header_parts.append(fr"{short} FT")
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    for method in methods:
        for i, sparsity in enumerate(sparsities_map[method]):
            row = f"{method if i == 0 else ''} & {sparsity}"
            for dataset_key, _ in datasets:
                key = (method, sparsity)
                if key in results and dataset_key in results[key]:
                    baseline = results[key][dataset_key]['baseline']
                    final = results[key][dataset_key]['final']
                    row += f" & {baseline:.3f} & {final:.3f}"
                else:
                    row += r" & -- & --"
            row += r" \\" 
            lines.append(row)
        if method != methods[-1]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)

def generate_latex_table_compact(results):
    """Generate a compact LaTeX table using arrow notation (base→ft) organized by sparsity."""
    # Dataset order and display names
    datasets = [
        ('counterfact', 'CounterFact'),
        ('ai2_arc', 'AI2-ARC'),
        ('wmdp_bio', 'WMDP Bio'),
        ('wmdp_cyber', 'WMDP Cyber'),
        ('wmdp_chem', 'WMDP Chem'),
    ]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Downstream task performance across sparsity levels (Base→FT).}")
    lines.append(r"\label{tab:pruned_results}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\hline")

    # Header
    header = r"\textbf{Method} & \textbf{Spar.} & " + " & ".join([r"\textbf{" + name + "}" for _, name in datasets]) + r" \\"
    lines.append(header)
    lines.append(r"\hline")

    # Base (0%) row
    row = "Base & 0\\%"
    for dataset_key, _ in datasets:
        key = ('Base', '0%')
        if key in results and dataset_key in results[key]:
            baseline = results[key][dataset_key]['baseline']
            final = results[key][dataset_key]['final']
            row += f" & {baseline:.2f}→{final:.2f}"
        else:
            row += r" & --"
    row += r" \\"
    lines.append(row)
    lines.append(r"\hline")

    # Pruning methods organized by sparsity
    methods = ['Random', 'Attribution', 'Group Lasso']
    sparsities = ['30%', '63%', '80%']
    
    for sparsity in sparsities:
        for method in methods:
            sparsity_latex = sparsity.replace('%', r'\%')
            row = f"{method} & {sparsity_latex}"
            key = (method, sparsity)
            for dataset_key, _ in datasets:
                if key in results and dataset_key in results[key]:
                    baseline = results[key][dataset_key]['baseline']
                    final = results[key][dataset_key]['final']
                    row += f" & {baseline:.2f}→{final:.2f}"
                else:
                    row += r" & --"
            row += r" \\"
            lines.append(row)
        if sparsity != sparsities[-1]:
            lines.append(r"\hline")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_latex_table_80(results, include_base=True):
    """Generate a minimal LaTeX table with only Base (0%) and 80% sparsity rows."""
    # Dataset order and display names (short labels)
    datasets = [
        ('counterfact', 'CF'),
        ('ai2_arc', 'ARC'),
        ('wmdp_bio', 'Bio'),
        ('wmdp_cyber', 'Cyber'),
        ('wmdp_chem', 'Chem'),
    ]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Downstream task performance (only 80\% sparsity; Base shown for reference).}")
    lines.append(r"\label{tab:pruned_results}")
    lines.append(r"\begin{tabular}{ll" + "rr" * len(datasets) + "}")
    lines.append(r"\toprule")

    # Header
    header_parts = [r"\textbf{Method}", r"\textbf{Sparsity}"]
    for _, short in datasets:
        header_parts.append(fr"{short} Base")
        header_parts.append(fr"{short} FT")
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    # Optional Base row
    if include_base:
        row = "Base & 0%"
        for dataset_key, _ in datasets:
            key = ('Base', '0%')
            if key in results and dataset_key in results[key]:
                baseline = results[key][dataset_key]['baseline']
                final = results[key][dataset_key]['final']
                row += f" & {baseline:.3f} & {final:.3f}"
            else:
                row += r" & -- & --"
        row += r" \\"
        lines.append(row)
        lines.append(r"\midrule")

    # Only 80% sparsity rows for pruned methods
    for method in ['Random', 'Attribution', 'Group Lasso']:
        row = f"{method} & 80%"
        key = (method, '80%')
        for dataset_key, _ in datasets:
            if key in results and dataset_key in results[key]:
                baseline = results[key][dataset_key]['baseline']
                final = results[key][dataset_key]['final']
                row += f" & {baseline:.3f} & {final:.3f}"
            else:
                row += r" & -- & --"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)

def print_summary_stats(results):
    """Print summary statistics."""
    print("\n=== Summary Statistics ===\n")
    
    datasets = ['counterfact', 'ai2_arc', 'wmdp_bio', 'wmdp_cyber', 'wmdp_chem']
    methods = ['Base', 'Random', 'Attribution', 'Group Lasso']
    sparsities_map = {
        'Base': ['0%'],
        'Random': ['30%', '63%', '80%'],
        'Attribution': ['30%', '63%', '80%'],
        'Group Lasso': ['30%', '63%', '80%']
    }
    
    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        print(f"{'Method':<12} {'Sparsity':<10} {'Baseline':<10} {'Final':<10} {'Δ':<10}")
        print("-" * 55)
        
        for method in methods:
            for sparsity in sparsities_map[method]:
                key = (method, sparsity)
                if key in results and dataset in results[key]:
                    baseline = results[key][dataset]['baseline']
                    final = results[key][dataset]['final']
                    delta = final - baseline
                    print(f"{method:<12} {sparsity:<10} {baseline:<10.3f} {final:<10.3f} {delta:+.3f}")
                else:
                    print(f"{method:<12} {sparsity:<10} {'--':<10} {'--':<10} {'--'}")


if __name__ == "__main__":
    import sys
    
    results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.environ.get('SCRATCH', '/tmp'),
        'iaifi_lab/Lab/ericjm/narrow/pruned_downstream_results'
    )
    
    print(f"Loading results from: {results_dir}")
    results = load_results(results_dir)
    
    if not results:
        print("No results found!")
        sys.exit(1)
    
    print(f"Loaded results for {len(results)} model configurations")
    
    # Print LaTeX table (compact arrow notation by default)
    print("\n" + "="*80)
    print("LATEX TABLE CODE (COMPACT ARROW NOTATION)")
    print("="*80 + "\n")
    compact_table = generate_latex_table_compact(results)
    print(compact_table)
    
    # Save compact table as default output (in this script's directory)
    output_file = os.path.join(os.path.dirname(__file__), "results_table.tex")
    with open(output_file, "w") as f:
        f.write(compact_table)
    print(f"\n\nSaved to: {output_file}")
    
    # Also write other table variants separately
    table_80 = generate_latex_table_80(results, include_base=True)
    output_80_file = os.path.join(os.path.dirname(__file__), "results_table_80only.tex")
    with open(output_80_file, "w") as f:
        f.write(table_80)
    print(f"Saved 80%-only table to: {output_80_file}")
    
    simple_table_full_sparsities = generate_latex_table_simple(results)
    simple_full_output_file = os.path.join(os.path.dirname(__file__), "results_table_simple_full.tex")
    with open(simple_full_output_file, "w") as f:
        f.write(simple_table_full_sparsities)
    print(f"Saved simplified full-sparsity table to: {simple_full_output_file}")

    full_table = generate_latex_table(results)
    full_output_file = os.path.join(os.path.dirname(__file__), "results_table_full.tex")
    with open(full_output_file, "w") as f:
        f.write(full_table)
    print(f"Saved full table to: {full_output_file}")

    # Print summary stats
    print_summary_stats(results)

