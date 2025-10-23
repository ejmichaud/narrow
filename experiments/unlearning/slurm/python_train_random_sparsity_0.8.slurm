#!/bin/bash
#SBATCH --job-name=py_random_sparsity_0.8
#SBATCH --partition=gpu
#SBATCH --mem=64GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 720
#SBATCH --output=slurm_logs/python_train_random_sparsity_0.8_%j.out
#SBATCH --error=slurm_logs/python_train_random_sparsity_0.8_%j.err

echo "Training random_sparsity_0.8 on Python code"
echo "Input model: /n/netscratch/iaifi_lab/Lab/ericjm/narrow/random_pruned/sparsity_0.8"
echo "Output: /n/netscratch/iaifi_lab/Lab/ericjm/narrow/python_trained/random_sparsity_0.8"
echo ""

# Set HF cache to SCRATCH
export HF_HOME="${SCRATCH}/iaifi_lab/Lab/ericjm/.cache/huggingface"

cd ..
/n/home04/ericjm/.conda/envs/narrow/bin/python train_on_python.py \
    --model_path "/n/netscratch/iaifi_lab/Lab/ericjm/narrow/random_pruned/sparsity_0.8" \
    --output_dir "/n/netscratch/iaifi_lab/Lab/ericjm/narrow/python_trained/random_sparsity_0.8" \
    --max_steps 10000 \
    --save_steps 2500 \
    --batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --max_length 1024 \
    --convert_to_variable_size

echo ""
echo "Job completed at $(date)"
