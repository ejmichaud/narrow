#!/bin/bash
#SBATCH --job-name=base-attr
#SBATCH --ntasks=1
#SBATCH --time=0-1:30:00
#SBATCH --output=/om2/user/ericjm/narrow/experiments/tuneprune7/base-attribution-abs/logs/slurm-%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=24GB
#SBATCH --mem=16GB
#SBATCH --array=1-26

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

echo "Running with sparsity=${sparsity}"

python /om2/user/ericjm/narrow/experiments/tuneprune7/prune-abs.py \
    --model_name "NousResearch/Llama-3.2-1B" \
    --pruning_strategy attribution \
    --max_length 512 \
    --batch_size 4 \
    --num_samples 2048 \
    --streaming \
    --sparsity "${sparsity}" \
    --output_dir /om2/user/ericjm/narrow/experiments/tuneprune7/base-attribution-abs/sparsity_${sparsity} \
    --num_test_samples 2048 \
    --eval_skip 2048
