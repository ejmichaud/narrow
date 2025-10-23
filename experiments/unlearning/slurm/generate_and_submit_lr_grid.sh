#!/bin/bash
# Grid search over learning rates for fine-tuning experiments

SCRATCH=${SCRATCH:-/n/netscratch}
TUNEPRUNE_DIR="${SCRATCH}/iaifi_lab/Lab/ericjm/narrow/tuneprune15-redo"
CHECKPOINT="checkpoint-70000"

# Learning rate grid
LR_GRID_COUNTERFACT=(1e-6 5e-6 1e-5 5e-5 1e-4)
LR_GRID_AI2_ARC=(1e-6 5e-6 1e-5 2e-5 5e-5)

echo "================================"
echo "Learning Rate Grid Search"
echo "================================"
echo ""
echo "CounterFact LRs: ${LR_GRID_COUNTERFACT[@]}"
echo "AI2-ARC LRs:     ${LR_GRID_AI2_ARC[@]}"
echo ""

# Create directories
mkdir -p slurm_logs
mkdir -p ../results

# Find all lambda directories
echo "Searching for models in ${TUNEPRUNE_DIR}..."
LAMBDA_DIRS=($(ls -d ${TUNEPRUNE_DIR}/lambda_*/checkpoint-70000 2>/dev/null))

if [ ${#LAMBDA_DIRS[@]} -eq 0 ]; then
    echo "WARNING: No tuneprune models found in ${TUNEPRUNE_DIR}"
    echo "Will only run base model experiments"
fi

echo "Found ${#LAMBDA_DIRS[@]} tuneprune models"
echo ""

JOB_IDS=()

# Helper function to format learning rate for filenames (replace . with p)
format_lr() {
    echo "$1" | sed 's/\./_/g' | sed 's/e-/e/'
}

# Submit base model jobs for each learning rate
echo "================================"
echo "Base Model Grid Search"
echo "================================"

for lr in "${LR_GRID_COUNTERFACT[@]}"; do
    lr_name=$(format_lr $lr)
    output_dir="../results/base_model_lr_${lr_name}/counterfact"
    mkdir -p "$output_dir"
    
    cat > base_counterfact_lr_${lr_name}.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=base_cf_lr${lr_name}
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 120
#SBATCH --output=slurm_logs/base_counterfact_lr${lr_name}_%j.out
#SBATCH --error=slurm_logs/base_counterfact_lr${lr_name}_%j.err

# Run experiment
cd ../counterfact
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval.py \\
    --model_path "NousResearch/Llama-3.2-1B" \\
    --output_dir "${output_dir}" \\
    --learning_rate ${lr}

echo "Job completed at \$(date)"
EOF
    
    JOB=$(sbatch --parsable base_counterfact_lr_${lr_name}.slurm)
    echo "  Base/CounterFact/lr=${lr}: Job ID ${JOB}"
    JOB_IDS+=($JOB)
done
echo ""

for lr in "${LR_GRID_AI2_ARC[@]}"; do
    lr_name=$(format_lr $lr)
    output_dir="../results/base_model_lr_${lr_name}/ai2_arc"
    mkdir -p "$output_dir"
    
    cat > base_ai2_arc_lr_${lr_name}.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=base_arc_lr${lr_name}
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 120
#SBATCH --output=slurm_logs/base_ai2_arc_lr${lr_name}_%j.out
#SBATCH --error=slurm_logs/base_ai2_arc_lr${lr_name}_%j.err

# Run experiment
cd ../ai2_arc
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval.py \\
    --model_path "NousResearch/Llama-3.2-1B" \\
    --output_dir "${output_dir}" \\
    --learning_rate ${lr}

echo "Job completed at \$(date)"
EOF
    
    JOB=$(sbatch --parsable base_ai2_arc_lr_${lr_name}.slurm)
    echo "  Base/AI2-ARC/lr=${lr}: Job ID ${JOB}"
    JOB_IDS+=($JOB)
done
echo ""

# Submit tuneprune model jobs for each learning rate
if [ ${#LAMBDA_DIRS[@]} -gt 0 ]; then
    for model_path in "${LAMBDA_DIRS[@]}"; do
        lambda_name=$(basename $(dirname $model_path))
        echo "================================"
        echo "${lambda_name} Grid Search"
        echo "================================"
        
        # CounterFact
        for lr in "${LR_GRID_COUNTERFACT[@]}"; do
            lr_name=$(format_lr $lr)
            output_dir="../results/${lambda_name}_lr_${lr_name}/counterfact"
            mkdir -p "$output_dir"
            
            cat > ${lambda_name}_counterfact_lr_${lr_name}.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=${lambda_name}_cf_lr${lr_name}
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 120
#SBATCH --output=slurm_logs/${lambda_name}_counterfact_lr${lr_name}_%j.out
#SBATCH --error=slurm_logs/${lambda_name}_counterfact_lr${lr_name}_%j.err

# Run experiment
cd ../counterfact
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval.py \\
    --model_path "${model_path}" \\
    --output_dir "${output_dir}" \\
    --learning_rate ${lr}

echo "Job completed at \$(date)"
EOF
            
            JOB=$(sbatch --parsable ${lambda_name}_counterfact_lr_${lr_name}.slurm)
            echo "  CounterFact/lr=${lr}: Job ID ${JOB}"
            JOB_IDS+=($JOB)
        done
        
        # AI2-ARC
        for lr in "${LR_GRID_AI2_ARC[@]}"; do
            lr_name=$(format_lr $lr)
            output_dir="../results/${lambda_name}_lr_${lr_name}/ai2_arc"
            mkdir -p "$output_dir"
            
            cat > ${lambda_name}_ai2_arc_lr_${lr_name}.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=${lambda_name}_arc_lr${lr_name}
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 120
#SBATCH --output=slurm_logs/${lambda_name}_ai2_arc_lr${lr_name}_%j.out
#SBATCH --error=slurm_logs/${lambda_name}_ai2_arc_lr${lr_name}_%j.err

# Run experiment
cd ../ai2_arc
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval.py \\
    --model_path "${model_path}" \\
    --output_dir "${output_dir}" \\
    --learning_rate ${lr}

echo "Job completed at \$(date)"
EOF
            
            JOB=$(sbatch --parsable ${lambda_name}_ai2_arc_lr_${lr_name}.slurm)
            echo "  AI2-ARC/lr=${lr}: Job ID ${JOB}"
            JOB_IDS+=($JOB)
        done
        echo ""
    done
fi

echo "================================"
echo "All ${#JOB_IDS[@]} jobs submitted!"
echo "================================"
echo ""
echo "Job IDs: ${JOB_IDS[@]}"
echo ""
echo "Monitor all jobs with:"
echo "  squeue -u \$USER"
echo "  watch -n 5 squeue -u \$USER"
echo ""
echo "View all logs with:"
echo "  tail -f slurm_logs/*_\$(date +%Y)*.out"
echo ""
echo "View results when complete:"
echo "  python ../view_results.py ../results"
echo ""



