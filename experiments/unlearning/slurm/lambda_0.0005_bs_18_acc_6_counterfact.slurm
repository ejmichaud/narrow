#!/bin/bash
#SBATCH --job-name=lambda_0.0005_bs_18_acc_6_cf
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 120
#SBATCH --output=slurm_logs/lambda_0.0005_bs_18_acc_6_counterfact_%j.out
#SBATCH --error=slurm_logs/lambda_0.0005_bs_18_acc_6_counterfact_%j.err

# Run experiment
cd ../counterfact
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval.py \
    --model_path "/n/netscratch/iaifi_lab/Lab/ericjm/narrow/tuneprune15-redo/lambda_0.0005_bs_18_acc_6/checkpoint-70000" \
    --output_dir "../results/lambda_0.0005_bs_18_acc_6/counterfact"

echo "Job completed at $(date)"
