#!/bin/bash
#SBATCH --job-name=prune14
#SBATCH --partition=tegmark
#SBATCH --ntasks=1
#SBATCH --time=4-00:00:00
#SBATCH --output=/om2/user/ericjm/narrow/experiments/tuneprune14/logs/slurm-%A_%a.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32GB

echo "Running with sparsity_lambda=0.0003, batch_size=18, accumulations=6"

# Run the pruning/finetuning script
python /om2/user/ericjm/narrow/tuneprune-autoadjust.py \
    --model_name NousResearch/Llama-3.2-1B \
    --output_dir /om/user/ericjm/results/narrow/tuneprune14/lambda_0.0003_bs_18_acc_6 \
    --sparsity_lambda 0.0003 \
    --regularizer "l1_of_l2_of_mlps" \
    --lr "2e-6" \
    --max_steps 360000 \
    --save_steps 5000 \
    --limit_checkpoints -1 \
    --batch_size 18 \
    --accumulations 6 \
    --use_streaming \
