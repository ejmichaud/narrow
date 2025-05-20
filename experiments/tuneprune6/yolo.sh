#!/bin/bash
#SBATCH --job-name=prune6
#SBATCH --partition tegmark
#SBATCH --ntasks=1
#SBATCH --time=5-00:00:00
#SBATCH --output=/om2/user/ericjm/narrow/experiments/tuneprune6/logs/slurm-%A_%a.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=8GB
#SBATCH --array=0

lambdas=(
  0.00008
)

# Pick the appropriate lambda based on SLURM_ARRAY_TASK_ID
sparsity_lambda=${lambdas[$SLURM_ARRAY_TASK_ID]}

echo "Running with sparsity_lambda=${sparsity_lambda}"

# Run the pruning/finetuning script
python /om2/user/ericjm/narrow/tuneprune.py \
    --model_name NousResearch/Llama-3.2-1B \
    --output_dir /om2/user/ericjm/narrow/experiments/tuneprune6/yolo_lambda_${sparsity_lambda} \
    --sparsity_lambda "${sparsity_lambda}" \
    --regularizer "l1_of_l2_of_mlps" \
    --lr "1e-6" \
    --max_steps 500000 \
    --save_steps 10000 \
    --batch_size 24 \
    --accumulations 2 \
    --use_streaming

