#!/bin/bash
#SBATCH --job-name=eval_base_cyber
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 120
#SBATCH --output=slurm_logs/eval_base_wmdp_cyber_%j.out
#SBATCH --error=slurm_logs/eval_base_wmdp_cyber_%j.err

export HF_HOME="${SCRATCH}/iaifi_lab/Lab/ericjm/.cache/huggingface"

cd ..
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval_lr_sweep.py \
    --model_path "NousResearch/Llama-3.2-1B" \
    --output_dir "/n/netscratch/iaifi_lab/Lab/ericjm/narrow/pruned_downstream_results/base_model/wmdp_cyber" \
    --dataset wmdp_cyber \
    --lr_sweep 1e-6 5e-6 1e-5 2e-5 5e-5

echo "Job completed at $(date)"
