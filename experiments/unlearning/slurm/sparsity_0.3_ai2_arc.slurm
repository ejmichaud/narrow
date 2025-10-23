#!/bin/bash
#SBATCH --job-name=sparsity_0.3_arc
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 120
#SBATCH --output=slurm_logs/sparsity_0.3_ai2_arc_%j.out
#SBATCH --error=slurm_logs/sparsity_0.3_ai2_arc_%j.err

# Run experiment
cd ../ai2_arc
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval.py \
    --model_path "/n/netscratch/iaifi_lab/Lab/ericjm/narrow/random_pruned/sparsity_0.3" \
    --output_dir "../results/sparsity_0.3/ai2_arc"

echo "Job completed at $(date)"
