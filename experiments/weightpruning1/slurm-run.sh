#!/bin/bash
#SBATCH --job-name=weightpruning1
#SBATCH --ntasks=1
#SBATCH --time=0-48:00:00
#SBATCH --output=/om2/user/ericjm/narrow/experiments/tuneprune5/logs/slurm-%A_%a.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=8GB
#SBATCH --array=0-3

lambdas=(
  0.0025
  0.0005
  0.001
  0.002
)

# Pick the appropriate lambda based on SLURM_ARRAY_TASK_ID
sparsity_lambda=${lambdas[$SLURM_ARRAY_TASK_ID]}

echo "Running with sparsity_lambda=${sparsity_lambda}"

# Run the pruning/finetuning script
python /om2/user/ericjm/narrow/tuneprune.py \
    --model_name NousResearch/Llama-3.2-1B \
    --output_dir /om2/user/ericjm/narrow/experiments/tuneprune5/lambda_${sparsity_lambda} \
    --sparsity_lambda "${sparsity_lambda}" \
    --regularizer "l1_of_l2_of_mlps" \
    --lr "3e-6" \
    --max_steps 60000 \
    --save_steps 10000 \
    --batch_size 18 \
    --use_streaming

