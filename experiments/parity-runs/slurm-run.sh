#!/bin/bash
#SBATCH --job-name=parity
#SBATCH --ntasks=1
#SBATCH --time=0-03:00:00
#SBATCH --output=/om/user/ericjm/results/narrow/parity-runs/logs/slurm-%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --array=0-11

python /om2/user/ericjm/narrow/experiments/parity-runs/eval.py $SLURM_ARRAY_TASK_ID
