#!/bin/bash
#SBATCH --job-name=distill00
#SBATCH --partition=iaifi_gpu
#SBATCH --account=iaifi_lab
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --output=/n/home04/ericjm/narrow/experiments/distill00/logs/slurm-%A_%a.out
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --array=0-3

# Define the different temperature values for distillation
TEMPERATURE_VALUES=(
    1.0 \
    2.0 \
    5.0 \
    10.0
)

mamba activate narrow

# Get the current temperature value based on array task ID
CURRENT_TEMP=${TEMPERATURE_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "Running distillation with temperature=${CURRENT_TEMP}, batch_size=4, accumulations=2"

# Run the distillation script
python $HOME/narrow/distill.py \
    --teacher_model_name NousResearch/Meta-Llama-3.1-8B \
    --student_model_name NousResearch/Llama-3.2-1B \
    --output_dir $SCRATCH/iaifi_lab/Lab/ericjm/narrow/distill00/temp_${CURRENT_TEMP} \
    --temperature ${CURRENT_TEMP} \
    --alpha 0.5 \
    --lr "2e-5" \
    --max_steps 100000 \
    --save_steps 2500 \
    --limit_checkpoints 3 \
    --batch_size 8 \
    --accumulations 4 \
    --use_streaming \
