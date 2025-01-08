#!/bin/bash
#SBATCH --job-name=prune0
#SBATCH --ntasks=1
#SBATCH --time=0-10:00:00
#SBATCH --output=/om2/user/ericjm/narrow/experiments/tuneprune0/logs/slurm-%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=24GB
#SBATCH --mem=8GB
#SBATCH --array=0-8

# List of 27 sparsity lambdas for the grid search:
lambdas=(
  0.000000001
  0.00000001
  0.0000001
  0.000001
  0.00001
  0.0001
  0.001
  0.01
  0.1
)

# Pick the appropriate lambda based on SLURM_ARRAY_TASK_ID
sparsity_lambda=${lambdas[$SLURM_ARRAY_TASK_ID]}

echo "Running with sparsity_lambda=${sparsity_lambda}"

# Run the pruning/finetuning script
python /om2/user/ericjm/narrow/tuneprune.py \
    --model_name NousResearch/Llama-3.2-1B \
    --output_dir /om2/user/ericjm/narrow/experiments/tuneprune0/lambda_${sparsity_lambda} \
    --sparsity_lambda "${sparsity_lambda}" \
    --max_steps 10000 \
    --save_steps 2000 \
    --use_streaming
