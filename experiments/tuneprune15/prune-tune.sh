#!/bin/bash
#SBATCH --job-name=tune-attr
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --time=0-0:30:00
#SBATCH --output=/n/home04/ericjm/narrow/experiments/tuneprune15/tune-attribution/logs/slurm-%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=16GB
#SBATCH --array=0-323

mkdir -p /n/home04/ericjm/narrow/experiments/tuneprune15/tune-attribution/logs

# Define the runs we'll use checkpoints from
runs=(
  "lambda_0.0003_bs_18_acc_6"
  "lambda_0.0005_bs_18_acc_6"
  "lambda_0.001_bs_18_acc_6"
)

# Define checkpoint names
checkpoint_names=(
  "checkpoint-10000"
  "checkpoint-30000"
  "checkpoint-50000"
  "checkpoint-70000"
)

# Define sparsity values
sparsities=(
  0.0
  0.03
  0.06
  0.09
  0.12
  0.15
  0.18
  0.21
  0.24
  0.27
  0.3
  0.33
  0.36
  0.39
  0.42
  0.45
  0.5
  0.55
  0.6
  0.65
  0.7
  0.75
  0.8
  0.85
  0.9
  0.95
  0.99
)

# Calculate indices from array task ID
# Total combinations per run = checkpoints × sparsities = 4 × 27 = 108
# Total combinations per checkpoint = sparsities = 27
run_idx=$((SLURM_ARRAY_TASK_ID / 108))
checkpoint_idx=$(((SLURM_ARRAY_TASK_ID % 108) / 27))
sparsity_idx=$((SLURM_ARRAY_TASK_ID % 27))

run=${runs[$run_idx]}
checkpoint_name=${checkpoint_names[$checkpoint_idx]}
sparsity=${sparsities[$sparsity_idx]}

# Create output directory path
output_dir="/n/home04/ericjm/narrow/experiments/tuneprune15/tune-attribution/${run}/${checkpoint_name}/sparsity_${sparsity}"

# Check if output directory already exists
if [ -d "${output_dir}" ]; then
    echo "Output directory already exists for run=${run}, checkpoint=${checkpoint_name}, sparsity=${sparsity}"
    exit 0
fi

echo "Running with run=${run}, checkpoint=${checkpoint_name}, sparsity=${sparsity}"

# Create parent directories if they don't exist
mkdir -p "$(dirname "${output_dir}")"

python /n/home04/ericjm/narrow/experiments/tuneprune15/prune.py \
    --model_name "$SCRATCH/iaifi_lab/Lab/ericjm/narrow/tuneprune15/${run}/${checkpoint_name}" \
    --pruning_strategy attribution \
    --max_length 512 \
    --batch_size 8 \
    --num_samples 2048 \
    --streaming \
    --sparsity "${sparsity}" \
    --output_dir "${output_dir}" \
    --num_test_samples 2048 \
    --eval_skip 2048
