#!/bin/bash
#SBATCH --job-name=prune15-dist
#SBATCH --partition=iaifi_gpu
#SBATCH --account=iaifi_lab
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:4
#SBATCH --output=/n/home04/ericjm/narrow/experiments/tuneprune15/logs/slurm-%j.out
#SBATCH --time=0-01:00:00
#SBATCH --ntasks=4
#SBATCH --mem=64GB

# Set lambda value for this distributed job
LAMBDA=0.0005

mamba activate narrow

echo "Running distributed training with sparsity_lambda=${LAMBDA}, batch_size=18, accumulations=4 on 4 GPUs"

# Run the distributed pruning/finetuning script with torchrun
torchrun --nproc_per_node=4 $HOME/narrow/tuneprune-distributed.py \
    --model_name NousResearch/Llama-3.2-1B \
    --output_dir $SCRATCH/iaifi_lab/Lab/ericjm/narrow/tuneprune15/lambda_${LAMBDA}_distributed \
    --sparsity_lambda ${LAMBDA} \
    --regularizer "l1_of_l2_of_mlps" \
    --lr "2e-6" \
    --max_steps 90000 \
    --save_steps 10000 \
    --limit_checkpoints -1 \
    --batch_size 21 \
    --accumulations 4 \
    --use_streaming \
