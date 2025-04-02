#!/bin/bash
#SBATCH --job-name=train_scratch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --output=/n/home04/ericjm/narrow/experiments/trainscratch00/logs/slurm-%A_%a.out
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=12GB
#SBATCH --array=4

# Define model size configurations for grid search
MODEL_CONFIGS=(
    "hidden_size=256 num_layers=4 num_heads=4 intermediate_size=1024"
    "hidden_size=512 num_layers=8 num_heads=8 intermediate_size=2048"
    "hidden_size=768 num_layers=12 num_heads=12 intermediate_size=3072"
    "hidden_size=1024 num_layers=16 num_heads=16 intermediate_size=4096"
    "hidden_size=1536 num_layers=24 num_heads=24 intermediate_size=6144"
)

# Get the current model config based on array task ID
CURRENT_CONFIG=${MODEL_CONFIGS[$SLURM_ARRAY_TASK_ID]}

# Split the configuration into tokens by replacing "=" with a space
read -r -a tokens <<< "$(echo $CURRENT_CONFIG | sed 's/=/ /g')"
# tokens now contains: [ "hidden_size" "256" "num_layers" "4" "num_heads" "4" "intermediate_size" "1024" ]

# Extract the values from the correct positions in the tokens array
HIDDEN_SIZE_VAL=${tokens[1]}
NUM_LAYERS_VAL=${tokens[3]}
NUM_HEADS_VAL=${tokens[5]}
INTERMEDIATE_SIZE_VAL=${tokens[7]}

echo "Running with model configuration: hidden_size=${HIDDEN_SIZE_VAL}, num_layers=${NUM_LAYERS_VAL}, num_heads=${NUM_HEADS_VAL}, intermediate_size=${INTERMEDIATE_SIZE_VAL}"

# Run the training script
python $HOME/narrow/trainscratch.py \
    --tokenizer_name "NousResearch/Meta-Llama-3.1-8B" \
    --hidden_size ${HIDDEN_SIZE_VAL} \
    --num_layers ${NUM_LAYERS_VAL} \
    --num_heads ${NUM_HEADS_VAL} \
    --intermediate_size ${INTERMEDIATE_SIZE_VAL} \
    --output_dir $SCRATCH/iaifi_lab/Lab/ericjm/narrow/trainscratch00/model_${HIDDEN_SIZE_VAL}_${NUM_LAYERS_VAL}_${NUM_HEADS_VAL} \
    --lr "5e-4" \
    --max_steps 100000 \
    --max_length 1024 \
    --batch_size 8 \
    --accumulations 8 \
    --logging_steps 5 \
    --save_steps 5000 \
    --limit_checkpoints 3 \
    --use_streaming
