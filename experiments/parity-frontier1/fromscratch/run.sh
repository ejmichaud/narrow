#!/bin/bash
#SBATCH --job-name=pscratch
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=0-2:00:00
#SBATCH --output=/om2/user/ericjm/narrow/experiments/parity-frontier1/fromscratch/logs/pscratch-%A_%a.out
#SBATCH --mem=8GB
#SBATCH --array=0-11

python /om2/user/ericjm/narrow/experiments/parity-frontier1/fromscratch/train.py $SLURM_ARRAY_TASK_ID

