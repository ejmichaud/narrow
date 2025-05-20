#!/bin/bash
#SBATCH --job-name=prune16
#SBATCH --partition=iaifi_gpu
#SBATCH --account=iaifi_lab
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --output=/n/home04/ericjm/narrow/experiments/tuneprune16/logs/slurm-%A_%a.out
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=20GB
#SBATCH --array=0-3

# Define the different sparsity lambda values
LAMBDA_VALUES=(
    0.0005 \
    0.001 \
    0.005 \
    0.01
)

mamba activate narrow

# Get the current lambda value based on array task ID
CURRENT_LAMBDA=${LAMBDA_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "Running with sparsity_lambda=${CURRENT_LAMBDA}, batch_size=8, accumulations=8"

# Run the pruning/finetuning script
python $HOME/narrow/tuneprune-autoadjust.py \
    --model_name NousResearch/Llama-3.2-1B \
    --output_dir $SCRATCH/iaifi_lab/Lab/ericjm/narrow/tuneprune16/lambda_${CURRENT_LAMBDA} \
    --sparsity_lambda ${CURRENT_LAMBDA} \
    --regularizer "l1_of_l2_of_mlps" \
    --lr "5e-5" \
    --max_steps 50000 \
    --save_steps 5000 \
    --limit_checkpoints -1 \
    --batch_size 8 \
    --accumulations 8 \
    --use_streaming \
