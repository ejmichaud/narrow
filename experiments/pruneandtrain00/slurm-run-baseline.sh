#!/bin/bash
#SBATCH --job-name=prune_train
#SBATCH --partition=iaifi_gpu
#SBATCH --account=iaifi_lab
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --output=/n/home04/ericjm/narrow/experiments/pruneandtrain00/logs/slurm-%A_%a.out
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=12GB
#SBATCH --array=0

# Create logs directory if it doesn't exist
mkdir -p /n/home04/ericjm/narrow/experiments/pruneandtrain00/logs

# Load environment
mamba activate narrow

# Define experiments as pairs of checkpoint path, display name, and sparsity level
# Format: "checkpoint_path display_name sparsity"
EXPERIMENTS=(
    "NousResearch/Llama-3.2-1B baseline 0.8"
)

# Get the current experiment
EXPERIMENT=(${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]})
MODEL_PATH=${EXPERIMENT[0]}
MODEL_NAME=${EXPERIMENT[1]}
SPARSITY=${EXPERIMENT[2]}

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running with model: $MODEL_PATH (display name: $MODEL_NAME)"
echo "Sparsity level: $SPARSITY"

# Create output directory
OUTPUT_DIR="$SCRATCH/iaifi_lab/Lab/ericjm/narrow/pruneandtrain00/${MODEL_NAME}_sparsity${SPARSITY}"
mkdir -p $OUTPUT_DIR

# Run the pruning and training script
python $HOME/narrow/prune_and_train.py \
    --model_name $MODEL_PATH \
    --dataset_name codeparrot/github-code \
    --max_length 1024 \
    --batch_size 18 \
    --accumulations 6 \
    --streaming \
    --sparsity $SPARSITY \
    --prune_samples 1024 \
    --train_skip 1024 \
    --max_steps 80000 \
    --lr 5e-5 \
    --mask_steps 1 \
    --output_dir $OUTPUT_DIR \
    --save_steps 10000 \
    --limit_checkpoints 3 \
    --logging_steps 5 \

# Save experiment metadata
cat > $OUTPUT_DIR/experiment_metadata.json << EOL
{
    "original_model_path": "$MODEL_PATH",
    "model_display_name": "$MODEL_NAME",
    "sparsity": $SPARSITY,
    "run_date": "$(date)",
    "slurm_job_id": "$SLURM_JOB_ID",
    "slurm_array_task_id": "$SLURM_ARRAY_TASK_ID"
}
EOL

echo "Experiment completed. Results saved to $OUTPUT_DIR"
