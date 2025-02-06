#!/bin/bash
#SBATCH --job-name=base-attr
#SBATCH --ntasks=1
#SBATCH --time=0-1:30:00
#SBATCH --output=/om2/user/ericjm/narrow/experiments/tuneprune5/base-attribution/logs/slurm-%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=24GB
#SBATCH --mem=16GB
#SBATCH --array=0-7

thresholds=(
  2e-10
  4e-10
  6e-10
  8e-10
  2e-9
  4e-9
  6e-9
  8e-9
)

# Pick the appropriate threshold based on SLURM_ARRAY_TASK_ID
threshold=${thresholds[$SLURM_ARRAY_TASK_ID]}

echo "Running with threshold=${threshold}"

python /om2/user/ericjm/narrow/experiments/tuneprune5/prune.py \
    --model_name "NousResearch/Llama-3.2-1B" \
    --pruning_strategy attribution \
    --max_length 512 \
    --batch_size 4 \
    --num_samples 2048 \
    --streaming \
    --importance_threshold "${threshold}" \
    --output_dir /om2/user/ericjm/narrow/experiments/tuneprune5/base-attribution/threshold_${threshold} \
    --num_test_samples 2048 \
    --eval_skip 2048
