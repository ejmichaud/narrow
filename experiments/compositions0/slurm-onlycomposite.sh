#!/bin/bash
#SBATCH --job-name=onlycomposite
#SBATCH --ntasks=1
#SBATCH --time=0-03:00:00
#SBATCH --output=/om/user/ericjm/results/narrow/compositions0/logs/slurm-%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --array=0-9

python /om2/user/ericjm/narrow/experiments/compositions0/eval-onlycomposite.py $SLURM_ARRAY_TASK_ID
