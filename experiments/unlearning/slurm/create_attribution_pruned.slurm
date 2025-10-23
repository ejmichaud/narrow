#!/bin/bash
#SBATCH --job-name=create_attr
#SBATCH --partition=gpu
#SBATCH --mem=64GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 180
#SBATCH --output=slurm_logs/create_attribution_%j.out
#SBATCH --error=slurm_logs/create_attribution_%j.err

echo "================================"
echo "Creating Attribution Pruned Models"
echo "================================"
echo ""

cd ..
/n/home04/ericjm/.conda/envs/narrow/bin/python create_attribution_pruned_models.py

echo ""
echo "================================"
echo "Attribution pruned models created!"
echo "================================"
echo "Job completed at $(date)"





