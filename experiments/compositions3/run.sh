#!/bin/bash
#SBATCH --job-name=comp3-width
#SBATCH --ntasks=1
#SBATCH --time=0-00:30:00
#SBATCH --output=/om2/user/ericjm/narrow/experiments/compositions3/logs/slurm-%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=24GB
#SBATCH --mem=8GB
#SBATCH --array=0-34

# Grid search over widths: 32, 64, 128, 256, 512, 1024, 2048
# and 5 seeds each

WIDTHS=(32 64 128 256 512 1024 2048)
SEEDS=(0 1 2 3 4)

# Calculate width and seed indices
WIDTH_IDX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 5))

WIDTH=${WIDTHS[$WIDTH_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "Running with width=$WIDTH and seed=$SEED"

python /om2/user/ericjm/narrow/experiments/compositions3/trainprunesave.py \
    --width $WIDTH \
    --seed $SEED \
    --steps 20000 \
    --verbose
