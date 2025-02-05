#!/bin/bash
#SBATCH --job-name=prune_from_pretrained_attribution10k_examples
#SBATCH --partition tegmark
#SBATCH --ntasks=1
#SBATCH --time=0-2:00:00
#SBATCH --output=./narrow/experiments/asher_prune_from_pretrained/logs/slurm-%A_%a.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=8GB
#SBATCH --array=0
#SBATCH --mail-user=asher577@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --verbose


# List of 27 sparsity lambdas for the grid search:
# lambdas=(
#   0.000001
#   0.000003
#   0.00001
#   0.00003
#   0.0001
# )

# Pick the appropriate lambda based on SLURM_ARRAY_TASK_ID
# sparsity_lambda=${lambdas[$SLURM_ARRAY_TASK_ID]}

echo "Running prune_from_pretrained.py"

# Run the pruning/finetuning script
python ./narrow/prune_from_pretrained.py 
    # --model_name NousResearch/Llama-3.2-1B \
    # --output_dir /om2/user/ericjm/narrow/experiments/tuneprune1/lambda_${sparsity_lambda} \
    # --sparsity_lambda "${sparsity_lambda}" \
    # --max_steps 40000 \
    # --save_steps 5000 \
    # --batch_size 8 \
    # --use_streaming

