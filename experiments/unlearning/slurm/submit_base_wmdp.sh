#!/bin/bash
# Submit base model evaluations for WMDP datasets

SCRATCH=${SCRATCH:-/n/netscratch}
OUTPUT_BASE="${SCRATCH}/iaifi_lab/Lab/ericjm/narrow/pruned_downstream_results"
BASE_MODEL="NousResearch/Llama-3.2-1B"

# Learning rate sweep
WMDP_LRS="1e-6 5e-6 1e-5 2e-5 5e-5"

echo "================================"
echo "Base Model WMDP Evaluations"
echo "================================"
echo ""

mkdir -p slurm_logs
JOB_IDS=()

# WMDP-Bio
output_dir="${OUTPUT_BASE}/base_model/wmdp_bio"
mkdir -p ${output_dir}

cat > eval_base_wmdp_bio.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=eval_base_bio
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 120
#SBATCH --output=slurm_logs/eval_base_wmdp_bio_%j.out
#SBATCH --error=slurm_logs/eval_base_wmdp_bio_%j.err

export HF_HOME="\${SCRATCH}/iaifi_lab/Lab/ericjm/.cache/huggingface"

cd ..
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval_lr_sweep.py \\
    --model_path "${BASE_MODEL}" \\
    --output_dir "${output_dir}" \\
    --dataset wmdp_bio \\
    --lr_sweep ${WMDP_LRS}

echo "Job completed at \$(date)"
EOF

JOB_BIO=$(sbatch --parsable eval_base_wmdp_bio.slurm)
echo "WMDP-Bio:   Job ID ${JOB_BIO}"
JOB_IDS+=($JOB_BIO)

# WMDP-Cyber
output_dir="${OUTPUT_BASE}/base_model/wmdp_cyber"
mkdir -p ${output_dir}

cat > eval_base_wmdp_cyber.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=eval_base_cyber
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 120
#SBATCH --output=slurm_logs/eval_base_wmdp_cyber_%j.out
#SBATCH --error=slurm_logs/eval_base_wmdp_cyber_%j.err

export HF_HOME="\${SCRATCH}/iaifi_lab/Lab/ericjm/.cache/huggingface"

cd ..
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval_lr_sweep.py \\
    --model_path "${BASE_MODEL}" \\
    --output_dir "${output_dir}" \\
    --dataset wmdp_cyber \\
    --lr_sweep ${WMDP_LRS}

echo "Job completed at \$(date)"
EOF

JOB_CYBER=$(sbatch --parsable eval_base_wmdp_cyber.slurm)
echo "WMDP-Cyber: Job ID ${JOB_CYBER}"
JOB_IDS+=($JOB_CYBER)

# WMDP-Chem
output_dir="${OUTPUT_BASE}/base_model/wmdp_chem"
mkdir -p ${output_dir}

cat > eval_base_wmdp_chem.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=eval_base_chem
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 120
#SBATCH --output=slurm_logs/eval_base_wmdp_chem_%j.out
#SBATCH --error=slurm_logs/eval_base_wmdp_chem_%j.err

export HF_HOME="\${SCRATCH}/iaifi_lab/Lab/ericjm/.cache/huggingface"

cd ..
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval_lr_sweep.py \\
    --model_path "${BASE_MODEL}" \\
    --output_dir "${output_dir}" \\
    --dataset wmdp_chem \\
    --lr_sweep ${WMDP_LRS}

echo "Job completed at \$(date)"
EOF

JOB_CHEM=$(sbatch --parsable eval_base_wmdp_chem.slurm)
echo "WMDP-Chem:  Job ID ${JOB_CHEM}"
JOB_IDS+=($JOB_CHEM)

echo ""
echo "================================"
echo "3 base model jobs submitted!"
echo "================================"
echo ""
echo "Job IDs: ${JOB_IDS[@]}"

