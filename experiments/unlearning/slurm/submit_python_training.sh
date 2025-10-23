#!/bin/bash
# Submit Python training jobs for all pruned models

SCRATCH=${SCRATCH:-/n/netscratch}
RANDOM_DIR="${SCRATCH}/iaifi_lab/Lab/ericjm/narrow/random_pruned"
ATTRIBUTION_DIR="${SCRATCH}/iaifi_lab/Lab/ericjm/narrow/attribution_pruned"
TUNEPRUNE_DIR="${SCRATCH}/iaifi_lab/Lab/ericjm/narrow/tuneprune_pruned"
OUTPUT_BASE="${SCRATCH}/iaifi_lab/Lab/ericjm/narrow/python_trained"

echo "================================"
echo "Python Training Jobs"
echo "================================"
echo ""

mkdir -p slurm_logs
JOB_IDS=()

# Find all pruned models
ALL_MODELS=()

# Random pruned models
for dir in $(ls -d ${RANDOM_DIR}/sparsity_* 2>/dev/null); do
    name="random_$(basename $dir)"
    ALL_MODELS+=("${dir}:${name}")
done

# Attribution pruned models
for dir in $(ls -d ${ATTRIBUTION_DIR}/sparsity_* 2>/dev/null); do
    name="attribution_$(basename $dir)"
    ALL_MODELS+=("${dir}:${name}")
done

# Tuneprune pruned models
for dir in $(ls -d ${TUNEPRUNE_DIR}/*_sparsity_* 2>/dev/null); do
    name="tuneprune_$(basename $dir)"
    ALL_MODELS+=("${dir}:${name}")
done

if [ ${#ALL_MODELS[@]} -eq 0 ]; then
    echo "ERROR: No pruned models found!"
    echo ""
    echo "Create them first with:"
    echo "  sbatch create_all_pruned_models.slurm"
    echo ""
    exit 1
fi

echo "Found ${#ALL_MODELS[@]} pruned models to train:"
for model_info in "${ALL_MODELS[@]}"; do
    IFS=':' read -r path name <<< "$model_info"
    echo "  - ${name}"
done
echo ""

# Generate and submit jobs for each model
for model_info in "${ALL_MODELS[@]}"; do
    IFS=':' read -r model_path model_name <<< "$model_info"
    
    output_dir="${OUTPUT_BASE}/${model_name}"
    
    echo "Submitting: ${model_name}..."
    
    # Create SLURM script
    cat > python_train_${model_name}.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=py_${model_name}
#SBATCH --partition=gpu
#SBATCH --mem=64GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 720
#SBATCH --output=slurm_logs/python_train_${model_name}_%j.out
#SBATCH --error=slurm_logs/python_train_${model_name}_%j.err

echo "Training ${model_name} on Python code"
echo "Input model: ${model_path}"
echo "Output: ${output_dir}"
echo ""

# Set HF cache to SCRATCH
export HF_HOME="\${SCRATCH}/iaifi_lab/Lab/ericjm/.cache/huggingface"

cd ..
/n/home04/ericjm/.conda/envs/narrow/bin/python train_on_python.py \\
    --model_path "${model_path}" \\
    --output_dir "${output_dir}" \\
    --max_steps 10000 \\
    --save_steps 2500 \\
    --batch_size 8 \\
    --gradient_accumulation_steps 8 \\
    --learning_rate 5e-5 \\
    --max_length 1024 \\
    --convert_to_variable_size

echo ""
echo "Job completed at \$(date)"
EOF
    
    # Submit job
    JOB_ID=$(sbatch --parsable python_train_${model_name}.slurm)
    echo "  Job ID: ${JOB_ID}"
    JOB_IDS+=($JOB_ID)
done

echo ""
echo "================================"
echo "All ${#JOB_IDS[@]} jobs submitted!"
echo "================================"
echo ""
echo "Job IDs: ${JOB_IDS[@]}"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo "  watch -n 5 squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f slurm_logs/python_train_*_\$(date +%Y)*.out"
echo ""
echo "Results will be saved to:"
echo "  ${OUTPUT_BASE}/random_sparsity_*/"
echo "  ${OUTPUT_BASE}/attribution_sparsity_*/"
echo "  ${OUTPUT_BASE}/tuneprune_lambda_*_sparsity_*/"
echo ""



