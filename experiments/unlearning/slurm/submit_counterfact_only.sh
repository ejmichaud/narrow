#!/bin/bash
# Submit only CounterFact evaluations with increased time limit

SCRATCH=${SCRATCH:-/n/netscratch}
RANDOM_DIR="${SCRATCH}/iaifi_lab/Lab/ericjm/narrow/random_pruned"
ATTRIBUTION_DIR="${SCRATCH}/iaifi_lab/Lab/ericjm/narrow/attribution_pruned"
TUNEPRUNE_DIR="${SCRATCH}/iaifi_lab/Lab/ericjm/narrow/tuneprune_pruned"
OUTPUT_BASE="${SCRATCH}/iaifi_lab/Lab/ericjm/narrow/pruned_downstream_results"

CF_LRS="1e-6 5e-6 1e-5 5e-5 1e-4"

echo "================================"
echo "CounterFact Evaluations (6h limit)"
echo "================================"
echo ""

mkdir -p slurm_logs
JOB_IDS=()

# Find all pruned models
ALL_MODELS=()

for dir in $(ls -d ${RANDOM_DIR}/sparsity_* 2>/dev/null); do
    name="random_$(basename $dir)"
    ALL_MODELS+=("${dir}:${name}")
done

for dir in $(ls -d ${ATTRIBUTION_DIR}/sparsity_* 2>/dev/null); do
    name="attribution_$(basename $dir)"
    ALL_MODELS+=("${dir}:${name}")
done

for dir in $(ls -d ${TUNEPRUNE_DIR}/*_sparsity_* 2>/dev/null); do
    name="tuneprune_$(basename $dir)"
    ALL_MODELS+=("${dir}:${name}")
done

echo "Found ${#ALL_MODELS[@]} models"
echo ""

for model_info in "${ALL_MODELS[@]}"; do
    IFS=':' read -r model_path model_name <<< "$model_info"
    
    output_dir="${OUTPUT_BASE}/${model_name}/counterfact"
    mkdir -p ${output_dir}
    
    cat > eval_${model_name}_counterfact.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=eval_${model_name}_cf
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 600
#SBATCH --output=slurm_logs/eval_${model_name}_counterfact_%j.out
#SBATCH --error=slurm_logs/eval_${model_name}_counterfact_%j.err

export HF_HOME="\${SCRATCH}/iaifi_lab/Lab/ericjm/.cache/huggingface"

cd ..
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval_lr_sweep.py \\
    --model_path "${model_path}" \\
    --output_dir "${output_dir}" \\
    --dataset counterfact \\
    --lr_sweep ${CF_LRS}

echo "Job completed at \$(date)"
EOF
    
    JOB_ID=$(sbatch --parsable eval_${model_name}_counterfact.slurm 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "${model_name}: Job ID ${JOB_ID}"
        JOB_IDS+=($JOB_ID)
    fi
done

echo ""
echo "================================"
echo "${#JOB_IDS[@]} CounterFact jobs submitted!"
echo "================================"

