#!/bin/bash
#SBATCH --job-name=prune15
#SBATCH --partition=iaifi_gpu
#SBATCH --account=iaifi_lab
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --output=/n/home04/ericjm/narrow/experiments/tuneprune15/logs/slurm-%A_%a.out
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --array=0-1

# Define the different sparsity lambda values
LAMBDA_VALUES=(
    0.002 \
    0.005
)

mamba activate narrow

# Get the current lambda value based on array task ID
CURRENT_LAMBDA=${LAMBDA_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "Running with sparsity_lambda=${CURRENT_LAMBDA}, batch_size=18, accumulations=6"

# Run the pruning/finetuning script
python $HOME/narrow/tuneprune-autoadjust.py \
    --model_name NousResearch/Llama-3.2-1B \
    --output_dir $SCRATCH/iaifi_lab/Lab/ericjm/narrow/tuneprune15/lambda_${CURRENT_LAMBDA}_bs_18_acc_6 \
    --sparsity_lambda ${CURRENT_LAMBDA} \
    --regularizer "l1_of_l2_of_mlps" \
    --lr "5e-6" \
    --max_steps 360000 \
    --save_steps 10000 \
    --limit_checkpoints -1 \
    --batch_size 18 \
    --accumulations 6 \
    --use_streaming \
