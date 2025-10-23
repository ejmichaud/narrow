#!/bin/bash
#SBATCH --job-name=base_counterfact
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 120
#SBATCH --output=slurm_logs/base_counterfact_%j.out
#SBATCH --error=slurm_logs/base_counterfact_%j.err

# Run experiment
cd ../counterfact
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval.py \
    --model_path "NousResearch/Llama-3.2-1B" \
    --output_dir "../results/base_model/counterfact"

echo "Job completed at $(date)"


