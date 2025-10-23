#!/bin/bash
# Submit downstream evaluation jobs for pruned models (without Python training)

SCRATCH=${SCRATCH:-/n/netscratch}
RANDOM_DIR="${SCRATCH}/iaifi_lab/Lab/ericjm/narrow/random_pruned"
ATTRIBUTION_DIR="${SCRATCH}/iaifi_lab/Lab/ericjm/narrow/attribution_pruned"
TUNEPRUNE_DIR="${SCRATCH}/iaifi_lab/Lab/ericjm/narrow/tuneprune_pruned"
OUTPUT_BASE="${SCRATCH}/iaifi_lab/Lab/ericjm/narrow/pruned_downstream_results"

# Learning rates for each dataset
CF_LRS="1e-6 5e-6 1e-5 5e-5 1e-4"
ARC_LRS="1e-6 5e-6 1e-5 2e-5 5e-5"
WMDP_LRS="1e-6 5e-6 1e-5 2e-5 5e-5"  # Same as ARC

echo "================================"
echo "Pruned Model Downstream Evaluation"
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
    exit 1
fi

echo "Found ${#ALL_MODELS[@]} pruned models to evaluate:"
for model_info in "${ALL_MODELS[@]}"; do
    IFS=':' read -r path name <<< "$model_info"
    echo "  - ${name}"
done
echo ""

# Submit jobs for each model × dataset
for model_info in "${ALL_MODELS[@]}"; do
    IFS=':' read -r model_path model_name <<< "$model_info"
    
    echo "Submitting: ${model_name}..."
    
    # CounterFact job
    output_dir="${OUTPUT_BASE}/${model_name}/counterfact"
    mkdir -p ${output_dir}
    
    cat > eval_${model_name}_counterfact.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=eval_${model_name}_cf
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 360
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
    
    JOB_CF=$(sbatch --parsable eval_${model_name}_counterfact.slurm 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "  CounterFact: Job ID ${JOB_CF}"
        JOB_IDS+=($JOB_CF)
    fi
    
    # AI2-ARC job
    output_dir="${OUTPUT_BASE}/${model_name}/ai2_arc"
    mkdir -p ${output_dir}
    
    cat > eval_${model_name}_ai2_arc.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=eval_${model_name}_arc
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 180
#SBATCH --output=slurm_logs/eval_${model_name}_ai2_arc_%j.out
#SBATCH --error=slurm_logs/eval_${model_name}_ai2_arc_%j.err

export HF_HOME="\${SCRATCH}/iaifi_lab/Lab/ericjm/.cache/huggingface"

cd ..
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval_lr_sweep.py \\
    --model_path "${model_path}" \\
    --output_dir "${output_dir}" \\
    --dataset ai2_arc \\
    --lr_sweep ${ARC_LRS}

echo "Job completed at \$(date)"
EOF
    
    JOB_ARC=$(sbatch --parsable eval_${model_name}_ai2_arc.slurm 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "  AI2-ARC:     Job ID ${JOB_ARC}"
        JOB_IDS+=($JOB_ARC)
    fi
    
    # WMDP-Bio job
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

cd ../wmdp_bio
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval.py \\
    --model_path "${model_path}" \\
    --output_dir "${output_dir}"

echo "Job completed at \$(date)"
EOF
    
    JOB_BIO=$(sbatch --parsable eval_${model_name}_wmdp_bio.slurm 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "  WMDP-Bio:    Job ID ${JOB_BIO}"
        JOB_IDS+=($JOB_BIO)
    fi
    
    # WMDP-Cyber job
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

cd ../wmdp_cyber
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval.py \\
    --model_path "${model_path}" \\
    --output_dir "${output_dir}"

echo "Job completed at \$(date)"
EOF
    
    JOB_CYBER=$(sbatch --parsable eval_${model_name}_wmdp_cyber.slurm 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "  WMDP-Cyber:  Job ID ${JOB_CYBER}"
        JOB_IDS+=($JOB_CYBER)
    fi
    
    # WMDP-Chem job
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

cd ../wmdp_chem
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval.py \\
    --model_path "${model_path}" \\
    --output_dir "${output_dir}"

echo "Job completed at \$(date)"
EOF
    
    JOB_CHEM=$(sbatch --parsable eval_${model_name}_wmdp_chem.slurm 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "  WMDP-Chem:   Job ID ${JOB_CHEM}"
        JOB_IDS+=($JOB_CHEM)
    fi
    
    echo ""
done

echo "================================"
echo "All ${#JOB_IDS[@]} jobs submitted!"
echo "================================"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER | grep eval"
echo ""
echo "View progress:"
echo "  cd slurm && ./monitor_jobs.sh"
echo ""
echo "Results will be saved to:"
echo "  ${OUTPUT_BASE}/"
echo ""

