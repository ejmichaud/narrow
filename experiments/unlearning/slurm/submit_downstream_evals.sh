#!/bin/bash
# Submit downstream evaluation jobs for all Python-trained checkpoints

SCRATCH=${SCRATCH:-/n/netscratch}
PYTHON_TRAINED_DIR="${SCRATCH}/iaifi_lab/Lab/ericjm/narrow/python_trained"
OUTPUT_BASE="${SCRATCH}/iaifi_lab/Lab/ericjm/narrow/downstream_results"

# Learning rates for each dataset
CF_LRS="1e-6 5e-6 1e-5 5e-5 1e-4"
ARC_LRS="1e-6 5e-6 1e-5 2e-5 5e-5"

echo "================================"
echo "Downstream Evaluation Jobs"
echo "================================"
echo ""

mkdir -p slurm_logs
JOB_IDS=()

# Find final checkpoints (step 10000) from Python training
CHECKPOINTS=()

for model_dir in $(find ${PYTHON_TRAINED_DIR} -maxdepth 1 -type d -name "*_sparsity_*" 2>/dev/null); do
    model_name=$(basename $model_dir)
    
    # Only evaluate the final checkpoint (step 10000)
    final_checkpoint="${model_dir}/checkpoint-10000"
    if [ -d "${final_checkpoint}" ]; then
        CHECKPOINTS+=("${final_checkpoint}:${model_name}_final")
    else
        echo "WARNING: Final checkpoint not found for ${model_name}"
    fi
done

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "ERROR: No checkpoints found in ${PYTHON_TRAINED_DIR}"
    echo ""
    echo "Train models on Python first with:"
    echo "  ./submit_python_training.sh"
    echo ""
    exit 1
fi

echo "Found ${#CHECKPOINTS[@]} checkpoints to evaluate"
echo ""

# Submit jobs for each checkpoint × dataset
for checkpoint_info in "${CHECKPOINTS[@]}"; do
    IFS=':' read -r checkpoint_path result_name <<< "$checkpoint_info"
    
    # CounterFact job
    output_dir="${OUTPUT_BASE}/${result_name}/counterfact"
    mkdir -p ${output_dir}
    
    cat > eval_${result_name}_counterfact.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=eval_cf_${result_name}
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 180
#SBATCH --output=slurm_logs/eval_${result_name}_counterfact_%j.out
#SBATCH --error=slurm_logs/eval_${result_name}_counterfact_%j.err

cd ..
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval_lr_sweep.py \\
    --model_path "${checkpoint_path}" \\
    --output_dir "${output_dir}" \\
    --dataset counterfact \\
    --lr_sweep ${CF_LRS}

echo "Job completed at \$(date)"
EOF
    
    JOB_ID=$(sbatch --parsable eval_${result_name}_counterfact.slurm 2>/dev/null)
    if [ $? -eq 0 ]; then
        JOB_IDS+=($JOB_ID)
    fi
    
    # AI2-ARC job
    output_dir="${OUTPUT_BASE}/${result_name}/ai2_arc"
    mkdir -p ${output_dir}
    
    cat > eval_${result_name}_ai2_arc.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=eval_arc_${result_name}
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 180
#SBATCH --output=slurm_logs/eval_${result_name}_ai2_arc_%j.out
#SBATCH --error=slurm_logs/eval_${result_name}_ai2_arc_%j.err

cd ..
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval_lr_sweep.py \\
    --model_path "${checkpoint_path}" \\
    --output_dir "${output_dir}" \\
    --dataset ai2_arc \\
    --lr_sweep ${ARC_LRS}

echo "Job completed at \$(date)"
EOF
    
    JOB_ID=$(sbatch --parsable eval_${result_name}_ai2_arc.slurm 2>/dev/null)
    if [ $? -eq 0 ]; then
        JOB_IDS+=($JOB_ID)
    fi
done

echo ""
echo "================================"
echo "All ${#JOB_IDS[@]} jobs submitted!"
echo "================================"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER | grep eval"
echo ""
echo "Results will be saved to:"
echo "  ${OUTPUT_BASE}/"
echo ""




