#!/bin/bash
#SBATCH --job-name=prune14
#SBATCH --ntasks=1
#SBATCH --time=0-01:00:00
#SBATCH --output=/om2/user/ericjm/narrow/experiments/tuneprune14/logs/slurm-%A_%a.out
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32GB
#SBATCH --array=0-1

# Define batch size and accumulation configurations
declare -a batch_sizes=(8 16)
declare -a accumulations=(2 1)

# Get the configuration based on SLURM_ARRAY_TASK_ID
batch_size=${batch_sizes[$SLURM_ARRAY_TASK_ID]}
accumulation=${accumulations[$SLURM_ARRAY_TASK_ID]}

echo "Running with batch_size=${batch_size}, accumulations=${accumulation}"

# Run the pruning/finetuning script
python /om2/user/ericjm/narrow/tuneprune-autoadjust.py \
    --model_name NousResearch/Llama-3.2-1B \
    --output_dir /om/user/ericjm/results/narrow/tuneprune14/bs_${batch_size}_acc_${accumulation} \
    --sparsity_lambda 0.0005 \
    --regularizer "l1_of_l2_of_mlps" \
    --lr "2e-6" \
    --max_steps 10000 \
    --save_steps 2500 \
    --limit_checkpoints 1 \
    --batch_size ${batch_size} \
    --accumulations ${accumulation} \
    --use_streaming \
