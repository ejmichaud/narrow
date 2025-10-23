#!/bin/bash
# Submit SLURM jobs only for random pruned models

SCRATCH=${SCRATCH:-/n/netscratch}
RANDOM_PRUNED_DIR="${SCRATCH}/iaifi_lab/Lab/ericjm/narrow/random_pruned"

echo "================================"
echo "Random Pruned Model Experiments"
echo "================================"
echo ""

# Create directories
mkdir -p slurm_logs
mkdir -p ../results

# Find all random pruned models
echo "Searching for random pruned models in ${RANDOM_PRUNED_DIR}..."
RANDOM_DIRS=($(ls -d ${RANDOM_PRUNED_DIR}/sparsity_* 2>/dev/null))

if [ ${#RANDOM_DIRS[@]} -eq 0 ]; then
    echo "ERROR: No random pruned models found in ${RANDOM_PRUNED_DIR}"
    echo ""
    echo "Create them first with:"
    echo "  python ../create_random_pruned_models.py"
    echo ""
    exit 1
fi

echo "Found ${#RANDOM_DIRS[@]} random pruned models:"
for dir in "${RANDOM_DIRS[@]}"; do
    random_name=$(basename $dir)
    echo "  - ${random_name}"
done
echo ""

# Submit jobs
echo "Submitting jobs..."
JOB_IDS=()

for model_path in "${RANDOM_DIRS[@]}"; do
    random_name=$(basename $model_path)
    echo "Submitting ${random_name} jobs..."
    
    # Create result directories
    mkdir -p ../results/${random_name}/{counterfact,ai2_arc}
    
    # Generate CounterFact job
    cat > ${random_name}_counterfact.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=${random_name}_cf
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 120
#SBATCH --output=slurm_logs/${random_name}_counterfact_%j.out
#SBATCH --error=slurm_logs/${random_name}_counterfact_%j.err

# Run experiment
cd ../counterfact
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval.py \\
    --model_path "${model_path}" \\
    --output_dir "../results/${random_name}/counterfact"

echo "Job completed at \$(date)"
EOF
    
    # Generate AI2-ARC job
    cat > ${random_name}_ai2_arc.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=${random_name}_arc
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 120
#SBATCH --output=slurm_logs/${random_name}_ai2_arc_%j.out
#SBATCH --error=slurm_logs/${random_name}_ai2_arc_%j.err

# Run experiment
cd ../ai2_arc
/n/home04/ericjm/.conda/envs/narrow/bin/python train_eval.py \\
    --model_path "${model_path}" \\
    --output_dir "../results/${random_name}/ai2_arc"

echo "Job completed at \$(date)"
EOF
    
    # Submit jobs
    JOB_CF=$(sbatch --parsable ${random_name}_counterfact.slurm)
    echo "  CounterFact: Job ID ${JOB_CF}"
    JOB_IDS+=($JOB_CF)
    
    JOB_ARC=$(sbatch --parsable ${random_name}_ai2_arc.slurm)
    echo "  AI2-ARC:     Job ID ${JOB_ARC}"
    JOB_IDS+=($JOB_ARC)
    
    echo ""
done

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
echo "  tail -f slurm_logs/sparsity_*_%j.out"
echo ""
echo "View results when complete:"
echo "  python ../view_results.py ../results"
echo ""

