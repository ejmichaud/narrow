# Learning Rate Grid Search Implementation

## Overview

Added comprehensive learning rate grid search functionality to the unlearning experiments.

## Changes Made

### 1. Updated Training Scripts
Both `counterfact/train_eval.py` and `ai2_arc/train_eval.py` now accept:
- `--learning_rate` argument (float)
- Default values maintained: 1e-5 (CounterFact), 5e-6 (AI2-ARC)

### 2. New Grid Search Submission Script
Created `slurm/generate_and_submit_lr_grid.sh`:
- Automatically generates SLURM jobs for each model × dataset × LR combination
- CounterFact LR grid: [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
- AI2-ARC LR grid: [1e-6, 5e-6, 1e-5, 2e-5, 5e-5]
- Results saved to: `results/{model}_lr_{lr}/{dataset}/results.json`

### 3. Analysis Script
Created `analyze_lr_grid.py`:
- Parses grid search results from directory structure
- Generates plots:
  - Final accuracy vs learning rate
  - Accuracy gain vs learning rate
- Identifies best learning rates for each model/dataset
- Usage: `python analyze_lr_grid.py results/`

### 4. Visualization Notebook
Updated `notebooks/downstream-visualize.ipynb`:
- Added grid search results loading
- Added learning rate curve plots (2×2 grid)
- Added best LR summary table
- Handles missing grid search data gracefully

## Usage

### Run Grid Search
```bash
cd experiments/unlearning/slurm
./generate_and_submit_lr_grid.sh
```

This will submit jobs for:
- Base model: 5 LRs × 2 datasets = 10 jobs
- Each tuneprune model: 5 LRs × 2 datasets = 10 jobs per model
- Total: ~40-50 jobs depending on number of tuneprune models

### Analyze Results
```bash
cd experiments/unlearning
python analyze_lr_grid.py results/
```

Or use the Jupyter notebook:
```bash
jupyter notebook notebooks/downstream-visualize.ipynb
```

## Directory Structure

Results are organized as:
```
results/
├── base_model_lr_1e_6/
│   ├── counterfact/results.json
│   └── ai2_arc/results.json
├── base_model_lr_5e_6/
│   ├── counterfact/results.json
│   └── ai2_arc/results.json
├── lambda_0.0003_bs_18_acc_6_lr_1e_5/
│   ├── counterfact/results.json
│   └── ai2_arc/results.json
...
```

## Learning Rate Format

Directory names use underscores instead of dots:
- `1e-5` → `1e_5`
- `5e-6` → `5e_6`

The analysis script automatically parses these back to float values.

## Benefits

1. **Systematic exploration**: Tests multiple LRs for each model
2. **Easy comparison**: Side-by-side plots for all models
3. **Automated**: Single script submits all jobs
4. **Organized**: Clear directory structure with LR in path
5. **Analysis tools**: Both CLI and notebook visualization



