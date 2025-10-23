#!/bin/bash
# Submit only WMDP evaluations for pruned models

SCRATCH=${SCRATCH:-/n/netscratch}
RANDOM_DIR="${SCRATCH}/iaifi_lab/Lab/ericjm/narrow/random_pruned"
ATTRIBUTION_DIR="${SCRATCH}/iaifi_lab/Lab/ericjm/narrow/attribution_pruned"
TUNEPRUNE_DIR="${SCRATCH}/iaifi_lab/Lab/ericjm/narrow/tuneprune_pruned"
OUTPUT_BASE="${SCRATCH}/iaifi_lab/Lab/ericjm/narrow/pruned_downstream_results"

# Learning rate sweep for WMDP
WMDP_LRS="1e-6 5e-6 1e-5 2e-5 5e-5"

echo "================================"
echo "WMDP Dataset Evaluations"
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

# Submit WMDP jobs for each model
for model_info in "${ALL_MODELS[@]}"; do
    IFS=':' read -r model_path model_name <<< "$model_info"
    
    echo "Submitting WMDP evals for: ${model_name}..."
    
    # Bio
    output_dir="${OUTPUT_BASE}/${model_name}/wmdp_bio"
    mkdir -p ${output_dir}
    
    cat > eval_${model_name}_wmdp_bio.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=eval_${model_name}_bio
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 180
#SBATCH --output=slurm_logs/eval_${model_name}_wmdp_bio_%j.out
#SBATCH --error=slurm_logs/eval_${model_name}_wmdp_bio_%j.err

export HF_HOME="\${SCRATCH}/iaifi_lab/Lab/ericjm/.cache/huggingface"

cd ..
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval_lr_sweep.py \\
    --model_path "${model_path}" \\
    --output_dir "${output_dir}" \\
    --dataset wmdp_bio \\
    --lr_sweep ${WMDP_LRS}

echo "Job completed at \$(date)"
EOF
    
    JOB_BIO=$(sbatch --parsable eval_${model_name}_wmdp_bio.slurm 2>/dev/null)
    [ $? -eq 0 ] && echo "  WMDP-Bio:    ${JOB_BIO}" && JOB_IDS+=($JOB_BIO)
    
    # Cyber
    output_dir="${OUTPUT_BASE}/${model_name}/wmdp_cyber"
    mkdir -p ${output_dir}
    
    cat > eval_${model_name}_wmdp_cyber.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=eval_${model_name}_cyber
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 180
#SBATCH --output=slurm_logs/eval_${model_name}_wmdp_cyber_%j.out
#SBATCH --error=slurm_logs/eval_${model_name}_wmdp_cyber_%j.err

export HF_HOME="\${SCRATCH}/iaifi_lab/Lab/ericjm/.cache/huggingface"

cd ..
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval_lr_sweep.py \\
    --model_path "${model_path}" \\
    --output_dir "${output_dir}" \\
    --dataset wmdp_cyber \\
    --lr_sweep ${WMDP_LRS}

echo "Job completed at \$(date)"
EOF
    
    JOB_CYBER=$(sbatch --parsable eval_${model_name}_wmdp_cyber.slurm 2>/dev/null)
    [ $? -eq 0 ] && echo "  WMDP-Cyber:  ${JOB_CYBER}" && JOB_IDS+=($JOB_CYBER)
    
    # Chem
    output_dir="${OUTPUT_BASE}/${model_name}/wmdp_chem"
    mkdir -p ${output_dir}
    
    cat > eval_${model_name}_wmdp_chem.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=eval_${model_name}_chem
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 180
#SBATCH --output=slurm_logs/eval_${model_name}_wmdp_chem_%j.out
#SBATCH --error=slurm_logs/eval_${model_name}_wmdp_chem_%j.err

export HF_HOME="\${SCRATCH}/iaifi_lab/Lab/ericjm/.cache/huggingface"

cd ..
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval_lr_sweep.py \\
    --model_path "${model_path}" \\
    --output_dir "${output_dir}" \\
    --dataset wmdp_chem \\
    --lr_sweep ${WMDP_LRS}

echo "Job completed at \$(date)"
EOF
    
    JOB_CHEM=$(sbatch --parsable eval_${model_name}_wmdp_chem.slurm 2>/dev/null)
    [ $? -eq 0 ] && echo "  WMDP-Chem:   ${JOB_CHEM}" && JOB_IDS+=($JOB_CHEM)
    
    echo ""
done

echo "================================"
echo "${#JOB_IDS[@]} WMDP jobs submitted!"
echo "================================"


