#!/bin/bash
#SBATCH --job-name=base-wgt
#SBATCH --ntasks=1
#SBATCH --time=0-0:30:00
#SBATCH --output=/om2/user/ericjm/narrow/experiments/tuneprune5/base-weight/logs/slurm-%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=24GB
#SBATCH --mem=16GB
#SBATCH --array=6-9

thresholds=(
  0.8
  0.9
  0.95
  1.0
  1.05
  1.1
  1.15
  1.2
  1.25
  1.3
  1.35
  1.4
  1.45
  1.5
)

# Pick the appropriate threshold based on SLURM_ARRAY_TASK_ID
threshold=${thresholds[$SLURM_ARRAY_TASK_ID]}

echo "Running with threshold=${threshold}"

python /om2/user/ericjm/narrow/experiments/tuneprune5/prune.py \
    --model_name "NousResearch/Llama-3.2-1B" \
    --pruning_strategy weight_norm \
    --max_length 512 \
    --batch_size 4 \
    --streaming \
    --pruning_threshold "${threshold}" \
    --output_dir /om2/user/ericjm/narrow/experiments/tuneprune5/base-weight/threshold_${threshold} \
    --num_test_samples 2048 \
    --eval_skip 2048
