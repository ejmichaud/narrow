#!/bin/bash
#SBATCH --job-name=eval_attribution_sparsity_0.3_arc
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 180
#SBATCH --output=slurm_logs/eval_attribution_sparsity_0.3_ai2_arc_%j.out
#SBATCH --error=slurm_logs/eval_attribution_sparsity_0.3_ai2_arc_%j.err

export HF_HOME="${SCRATCH}/iaifi_lab/Lab/ericjm/.cache/huggingface"

cd ..
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval_lr_sweep.py \
    --model_path "/n/netscratch/iaifi_lab/Lab/ericjm/narrow/attribution_pruned/sparsity_0.3" \
    --output_dir "/n/netscratch/iaifi_lab/Lab/ericjm/narrow/pruned_downstream_results/attribution_sparsity_0.3/ai2_arc" \
    --dataset ai2_arc \
    --lr_sweep 1e-6 5e-6 1e-5 2e-5 5e-5

echo "Job completed at $(date)"
