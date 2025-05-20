#!/bin/bash
#SBATCH --job-name=prune10
#SBATCH --ntasks=1
#SBATCH --time=0-48:00:00
#SBATCH --output=/om2/user/ericjm/narrow/experiments/tuneprune10/logs/slurm-%A_%a.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=8GB
#SBATCH --array=0-2

lambdas=(
  0.0005
  0.001
  0.003
)

# Pick the appropriate lambda based on SLURM_ARRAY_TASK_ID
sparsity_lambda=${lambdas[$SLURM_ARRAY_TASK_ID]}

echo "Running with sparsity_lambda=${sparsity_lambda}"

# Run the pruning/finetuning script
python /om2/user/ericjm/narrow/tuneprune-autoadjust.py \
    --model_name NousResearch/Llama-3.2-1B \
    --output_dir /om2/user/ericjm/narrow/experiments/tuneprune10/lambda_${sparsity_lambda} \
    --sparsity_lambda "${sparsity_lambda}" \
    --regularizer "l1_of_l2_of_mlps" \
    --lr "3e-6" \
    --max_steps 240000 \
    --save_steps 2500 \
    --batch_size 18 \
    --accumulations 2 \
    --use_streaming \
    --use_adaptive_reg \
    --reg_warmup_steps 1000 \
    --adjust_steps 200 \
    --alpha 0.995 \
    --inc_factor 1.35 \
    --dec_factor 0.8
