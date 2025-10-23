#!/bin/bash
#SBATCH --job-name=create_pruned
#SBATCH --partition=gpu
#SBATCH --mem=64GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 180
#SBATCH --output=slurm_logs/create_pruned_%j.out
#SBATCH --error=slurm_logs/create_pruned_%j.err

echo "================================"
echo "Creating All Pruned Models"
echo "================================"
echo ""

cd ..

# 1. Create random pruned models
echo "Creating random pruned models..."
/n/home04/ericjm/.conda/envs/narrow/bin/python create_random_pruned_models.py

echo ""
echo "---"
echo ""

# 2. Create attribution pruned models
echo "Creating attribution pruned models..."
/n/home04/ericjm/.conda/envs/narrow/bin/python create_attribution_pruned_models.py

echo ""
echo "---"
echo ""

# 3. Create tuneprune pruned models
echo "Creating tuneprune pruned models..."
/n/home04/ericjm/.conda/envs/narrow/bin/python create_tuneprune_pruned_models.py

echo ""
echo "================================"
echo "All pruned models created!"
echo "================================"
echo ""
echo "Models saved to \$SCRATCH/iaifi_lab/Lab/ericjm/narrow/:"
echo "  - random_pruned/sparsity_{0.3,0.63,0.8}/"
echo "  - attribution_pruned/sparsity_{0.3,0.63,0.8}/"
echo "  - tuneprune_pruned/lambda_*_sparsity_*/"
echo ""
echo "Job completed at $(date)"






