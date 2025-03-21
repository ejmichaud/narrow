#!/bin/bash
#SBATCH --job-name=tune-attr
#SBATCH --account=iaifi_lab
#SBATCH --partition=iaifi_gpu
#SBATCH --ntasks=1
#SBATCH --time=0-0:30:00
#SBATCH --output=/n/home04/ericjm/narrow/experiments/tuneprune15/tune-attribution/logs/slurm-%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=16GB
#SBATCH --array=0-26

mkdir -p /n/home04/ericjm/narrow/experiments/tuneprune15/tune-attribution/logs

# Define checkpoint name as a variable for easy changing
checkpoint_name="checkpoint-70000"

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

# Pick the appropriate threshold based on SLURM_ARRAY_TASK_ID
sparsity=${sparsities[$SLURM_ARRAY_TASK_ID]}

# check if output directory already exists
if [ -d "/n/home04/ericjm/narrow/experiments/tuneprune15/tune-attribution/${checkpoint_name}/sparsity_${sparsity}" ]; then
    echo "Output directory already exists for sparsity=${sparsity}"
    exit 0
fi

echo "Running with sparsity=${sparsity}"

python /n/home04/ericjm/narrow/experiments/tuneprune15/prune.py \
    --model_name "$SCRATCH/iaifi_lab/Lab/ericjm/narrow/tuneprune15/lambda_0.0003_bs_18_acc_6/${checkpoint_name}" \
    --pruning_strategy attribution \
    --max_length 512 \
    --batch_size 8 \
    --num_samples 2048 \
    --streaming \
    --sparsity "${sparsity}" \
    --output_dir /n/home04/ericjm/narrow/experiments/tuneprune15/tune-attribution/${checkpoint_name}/sparsity_${sparsity} \
    --num_test_samples 2048 \
    --eval_skip 2048
