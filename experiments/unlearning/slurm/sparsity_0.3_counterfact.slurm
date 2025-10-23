#!/bin/bash
#SBATCH --job-name=sparsity_0.3_cf
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 120
#SBATCH --output=slurm_logs/sparsity_0.3_counterfact_%j.out
#SBATCH --error=slurm_logs/sparsity_0.3_counterfact_%j.err

# Run experiment
cd ../counterfact
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval.py \
    --model_path "/n/netscratch/iaifi_lab/Lab/ericjm/narrow/random_pruned/sparsity_0.3" \
    --output_dir "../results/sparsity_0.3/counterfact"

echo "Job completed at $(date)"
