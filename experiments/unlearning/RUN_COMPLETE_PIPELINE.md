# Complete Unlearning Experiment Pipeline

This document describes the complete experimental workflow for testing whether pruned models can robustly "unlearn" downstream tasks.

## Overview

We test three pruning methods (random, attribution-based, tuneprune L1-regularized) at three sparsity levels (30%, 63%, 80%). Each model is:
1. **Pruned** to target sparsity
2. **Trained on Python code** for 10k steps (with checkpoints every 2.5k steps)
3. **Evaluated on downstream tasks** (CounterFact, AI2-ARC) before and after fine-tuning

Total: **9 initial models → 45 checkpoints → ~90 downstream evaluations** (45 × 2 datasets)

## Step-by-Step Execution

### Step 1: Create All Pruned Models (9 models)

```bash
cd /n/home04/ericjm/narrow/experiments/unlearning/slurm
sbatch create_all_pruned_models.slurm
```

**What this does:**
- Creates 3 random pruned models (30%, 63%, 80%)
- Creates 3 attribution pruned models (based on Python code importance)
- Creates 3 tuneprune pruned models (from lambda checkpoints)

**Time:** ~2 hours

**Output locations:**
- `$SCRATCH/iaifi_lab/Lab/ericjm/narrow/random_pruned/sparsity_{0.3,0.63,0.8}/`
- `$SCRATCH/iaifi_lab/Lab/ericjm/narrow/attribution_pruned/sparsity_{0.3,0.63,0.8}/`
- `$SCRATCH/iaifi_lab/Lab/ericjm/narrow/tuneprune_pruned/lambda_*_sparsity_*/`

---

### Step 2: Train All Models on Python Code (9 → 45 models)

```bash
cd /n/home04/ericjm/narrow/experiments/unlearning/slurm
./submit_python_training.sh
```

**What this does:**
- Converts each pruned model to VariableSizeLlamaForCausalLM (physically removes neurons)
- Trains on `codeparrot/github-code` for 10k steps
- Saves checkpoints at steps 0, 2500, 5000, 7500, 10000
- No masking needed (neurons physically removed)

**Time:** ~6 hours per model (9 jobs × 6 hours)

**Output locations:**
- `$SCRATCH/iaifi_lab/Lab/ericjm/narrow/python_trained/random_sparsity_*/checkpoint-{0,2500,5000,7500,10000}/`
- `$SCRATCH/iaifi_lab/Lab/ericjm/narrow/python_trained/attribution_sparsity_*/checkpoint-{0,2500,5000,7500,10000}/`
- `$SCRATCH/iaifi_lab/Lab/ericjm/narrow/python_trained/tuneprune_lambda_*_sparsity_*/checkpoint-{0,2500,5000,7500,10000}/`

**Expected checkpoints:** 45 total (9 models × 5 checkpoints)

---

### Step 3: Evaluate Final Checkpoints on Downstream Tasks (18 evaluations)

```bash
cd /n/home04/ericjm/narrow/experiments/unlearning/slurm
./submit_downstream_evals.sh
```

**What this does:**
- For each model's final checkpoint (9 checkpoints at step 10000):
  - Evaluate baseline accuracy (no fine-tuning)
  - Fine-tune with LR sweep:
    - CounterFact: [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
    - AI2-ARC: [1e-6, 5e-6, 1e-5, 2e-5, 5e-5]
  - Report best accuracy from sweep

**Time:** ~3 hours per evaluation (18 jobs × 3 hours)

**Jobs:** 18 total (9 final checkpoints × 2 datasets)

**Output locations:**
- `$SCRATCH/iaifi_lab/Lab/ericjm/narrow/downstream_results/*/`

---

### Step 4: Analyze Results

```bash
cd /n/home04/ericjm/narrow/experiments/unlearning
python analyze_unlearning_results.py
```

This will create plots and tables comparing:
- **Baseline accuracy** (before downstream fine-tuning) vs Python training steps
- **Best fine-tuned accuracy** vs Python training steps
- Comparison across pruning methods (random vs attribution vs tuneprune)
- Effect of sparsity level

---

## Key Questions Being Tested

1. **Does Python training help or hurt downstream task performance?**
   - Compare checkpoint-0 vs checkpoint-10000

2. **Do pruned models "unlearn" downstream tasks more robustly?**
   - Higher baseline = still remembers
   - Lower baseline = successfully unlearned
   - Can they re-learn with fine-tuning?

3. **Does pruning method matter?**
   - Random vs attribution vs tuneprune
   - Which creates most robust unlearning?

4. **Does sparsity level matter?**
   - 30% vs 63% vs 80%
   - More aggressive pruning = better unlearning?

---

## Directory Structure

```
$SCRATCH/iaifi_lab/Lab/ericjm/narrow/
├── random_pruned/
│   ├── sparsity_0.3/
│   ├── sparsity_0.63/
│   └── sparsity_0.8/
├── attribution_pruned/
│   ├── sparsity_0.3/
│   ├── sparsity_0.63/
│   └── sparsity_0.8/
├── tuneprune_pruned/
│   ├── lambda_0.0003_bs_18_acc_6_sparsity_0.3/
│   ├── lambda_0.0005_bs_18_acc_6_sparsity_0.63/
│   └── lambda_0.001_bs_18_acc_6_sparsity_0.8/
├── python_trained/
│   ├── random_sparsity_0.3/
│   │   ├── checkpoint-2500/
│   │   ├── checkpoint-5000/
│   │   ├── checkpoint-7500/
│   │   └── checkpoint-10000/
│   ├── ... (8 more model directories)
└── downstream_results/
    ├── random_sparsity_0.3_step0/
    │   ├── counterfact/results.json
    │   └── ai2_arc/results.json
    ├── random_sparsity_0.3_checkpoint-2500/
    │   ├── counterfact/results.json
    │   └── ai2_arc/results.json
    └── ... (43 more checkpoint result directories)
```

## Monitoring Progress

```bash
# Monitor all jobs
watch -n 5 squeue -u $USER

# Check Python training progress
tail -f $SCRATCH/iaifi_lab/Lab/ericjm/narrow/python_trained/*/training_log.txt

# Check downstream eval progress
ls -ltr $SCRATCH/iaifi_lab/Lab/ericjm/narrow/downstream_results/*/*/results.json
```

## Total Resource Usage

- **Jobs:** ~104 total (1 creation + 9 Python training + 90 downstream evals + 4 baseline)
- **Compute time:** ~500 GPU-hours (assuming A100)
- **Storage:** ~100GB for all checkpoints




