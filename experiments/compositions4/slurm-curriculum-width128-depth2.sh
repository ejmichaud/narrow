#!/bin/bash
#SBATCH --job-name=curr-d2
#SBATCH --ntasks=1
#SBATCH --time=0-03:00:00
#SBATCH --output=/om2/user/ericjm/narrow/experiments/compositions4/logs/slurm-%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --array=23-26

# Grid search:
# - depth=2, seeds=0,1,2,3,4,5,6,7,8,9

DEPTH=2
SEED=$SLURM_ARRAY_TASK_ID

echo "Running with depth=$DEPTH and seed=$SEED"

python /om2/user/ericjm/narrow/experiments/compositions4/train.py \
    --width 128 \
    --depth $DEPTH \
    --seed $SEED \
    --codes "[[0], [1], [2], [3], [0, 1, 2, 3]]" \
    --samples-per-task 2000 \
    --steps 200000 \
    --verbose \
    --save-dir /om2/user/ericjm/narrow/experiments/compositions4/results/curriculum-depth$DEPTH-width128-seed$SEED
