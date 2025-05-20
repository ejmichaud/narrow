#!/bin/bash
#SBATCH --job-name=prune13
#SBATCH --ntasks=1
#SBATCH --time=0-48:00:00
#SBATCH --output=/om2/user/ericjm/narrow/experiments/tuneprune13/logs/slurm-%A_%a.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32GB
#SBATCH --array=0-2

lambdas=(
  0.00001
  0.0001
  0.001
)

# Pick the appropriate lambda based on SLURM_ARRAY_TASK_ID
sparsity_lambda=${lambdas[$SLURM_ARRAY_TASK_ID]}

echo "Running with sparsity_lambda=${sparsity_lambda}"

# Run the pruning/finetuning script
python /om2/user/ericjm/narrow/tuneprune-autoadjust.py \
    --model_name NousResearch/Llama-3.2-1B \
    --output_dir /om2/user/ericjm/narrow/experiments/tuneprune13/lambda_${sparsity_lambda} \
    --sparsity_lambda "${sparsity_lambda}" \
    --regularizer "group_lasso_residual_stream" \
    --lr "1e-6" \
    --max_steps 240000 \
    --limit_checkpoints -1 \
    --save_steps 2500 \
    --batch_size 18 \
    --accumulations 2 \
    --use_streaming \
    --use_adaptive_reg \
    --reg_warmup_steps 4000 \
    --adjust_steps 500 \
    --alpha 0.9999 \
    --inc_factor 1.35 \
    --dec_factor 0.8
