#!/bin/bash
#SBATCH --job-name=curr-sd2
#SBATCH --ntasks=1
#SBATCH --time=0-03:00:00
#SBATCH --output=/om2/user/ericjm/narrow/experiments/compositions2/logs/slurm-%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --array=1-15

# Grid search:
# - For indices 0-7: depth=2, seeds=0,1,2,3,4,5,6,7
# - For indices 8-15: depth=3, seeds=0,1,2,3,4,5,6,7

if [ $SLURM_ARRAY_TASK_ID -lt 8 ]; then
    # Indices 0-7: depth=2
    DEPTH=2
    SEED=$SLURM_ARRAY_TASK_ID
else
    # Indices 8-15: depth=3
    DEPTH=3
    SEED=$((SLURM_ARRAY_TASK_ID - 8))
fi

echo "Running with depth=$DEPTH and seed=$SEED"

python /om2/user/ericjm/narrow/experiments/compositions2/train-curriculum-skilldepth2.py \
    --width 128 \
    --depth $DEPTH \
    --seed $SEED \
    --verbose \
    --save-dir /om2/user/ericjm/narrow/experiments/compositions2/results/curriculum-skilldepth2-depth$DEPTH-seed$SEED
