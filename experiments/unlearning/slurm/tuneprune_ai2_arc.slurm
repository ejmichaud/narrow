#!/bin/bash
#SBATCH --job-name=tp_ai2_arc
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 120
#SBATCH --output=slurm_logs/tuneprune_ai2_arc_%j.out
#SBATCH --error=slurm_logs/tuneprune_ai2_arc_%j.err

# Run experiment
cd ../ai2_arc
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval.py \
    --model_path "${SCRATCH}/iaifi_lab/Lab/ericjm/narrow/tuneprune15-redo/lambda_0.0005_bs_18_acc_6/checkpoint-70000" \
    --output_dir "../results/tuneprune_model/ai2_arc"

echo "Job completed at $(date)"


