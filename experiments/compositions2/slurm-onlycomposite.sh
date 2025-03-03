#!/bin/bash
#SBATCH --job-name=onlycomp
#SBATCH --ntasks=1
#SBATCH --time=0-03:00:00
#SBATCH --output=/om2/user/ericjm/narrow/experiments/compositions2/logs/slurm-%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --array=0-11

# Grid search:
# - For indices 0-3: depth=2, seeds=0,1,2,3
# - For indices 4-7: depth=3, seeds=0,1,2,3
# - For indices 8-11: depth=4, seeds=0,1,2,3

if [ $SLURM_ARRAY_TASK_ID -lt 4 ]; then
    # Indices 0-3: depth=2
    DEPTH=2
    SEED=$SLURM_ARRAY_TASK_ID
elif [ $SLURM_ARRAY_TASK_ID -lt 8 ]; then
    # Indices 4-7: depth=3
    DEPTH=3
    SEED=$((SLURM_ARRAY_TASK_ID - 4))
else
    # Indices 8-11: depth=4
    DEPTH=4
    SEED=$((SLURM_ARRAY_TASK_ID - 8))
fi

echo "Running with depth=$DEPTH and seed=$SEED"

python /om2/user/ericjm/narrow/experiments/compositions2/train-onlycomposite.py \
    --width 128 \
    --depth $DEPTH \
    --seed $SEED \
    --samples-per-task 8000 \
    --steps 500000 \
    --verbose \
    --save-dir /om2/user/ericjm/narrow/experiments/compositions2/results/onlycomposite-depth$DEPTH-seed$SEED
