#!/bin/bash
#SBATCH --job-name=base_ai2_arc
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 120
#SBATCH --output=slurm_logs/base_ai2_arc_%j.out
#SBATCH --error=slurm_logs/base_ai2_arc_%j.err

# Run experiment
cd ../ai2_arc
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval.py \
    --model_path "NousResearch/Llama-3.2-1B" \
    --output_dir "../results/base_model/ai2_arc"

echo "Job completed at $(date)"


