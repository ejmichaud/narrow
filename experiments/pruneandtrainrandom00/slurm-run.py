#!/usr/bin/env python3
import os
import subprocess

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=prune_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --output=/n/home04/ericjm/narrow/experiments/pruneandtrainrandom00/logs/slurm-%A_%a.out
#SBATCH --time={time_limit}
#SBATCH --ntasks=1
#SBATCH --mem=12GB

# Load environment
mamba activate narrow

# Create output directory
mkdir -p {output_dir}

python $HOME/narrow/prune_and_train_random.py \
    --model_name {model_name} \
    --dataset_name codeparrot/github-code \
    --max_length {max_length} \
    --batch_size {batch_size} \
    --accumulations {accumulations} \
    --streaming \
    --output_dir {output_dir} \
    --neuron_sparsity {neuron_sparsity} \
    --residual_sparsity {residual_sparsity} \
    --prune_samples {prune_samples} \
    --train_skip {train_skip} \
    --max_steps {max_steps} \
    --lr {lr} \
    --mask_steps {mask_steps} \
    --eval_steps {eval_steps} \
    --save_steps {save_steps} \
    --limit_checkpoints {limit_checkpoints} \
    --logging_steps {logging_steps} \
    --warmup_steps {warmup_steps} \

# Save experiment metadata
cat > {output_dir}/experiment_metadata.json << EOL
{{
    "model_name": "{model_name}",
    "neuron_sparsity": {neuron_sparsity},
    "residual_sparsity": {residual_sparsity},
    "run_date": "$(date)",
    "slurm_job_id": "$SLURM_JOB_ID"
}}
EOL
"""

def main():
    # Base model to use
    model_name = "NousResearch/Llama-3.2-1B"
    
    # Define specific combinations of (neuron_sparsity, residual_sparsity)
    sparsity_configs = [
        (0.5, 0.2),
        (0.8, 0.5),
        (0.9, 0.9),
    ]
    
    # Common training parameters
    training_params = {
        "time_limit": "1-00:00:00",
        "max_length": 1024,
        "batch_size": 8,
        "accumulations": 8,
        "prune_samples": 1024,
        "train_skip": 1024,
        "max_steps": 20000,
        "lr": "5e-5",
        "mask_steps": 1,
        "eval_steps": 500,
        "save_steps": 2500,
        "limit_checkpoints": 3,
        "logging_steps": 5,
        "warmup_steps": 1000,
    }

    # Get SCRATCH directory from environment
    SCRATCH = os.environ.get('SCRATCH')
    if not SCRATCH:
        raise ValueError("SCRATCH environment variable not set")
    
    # Create logs directory
    os.makedirs("/n/home04/ericjm/narrow/experiments/pruneandtrainrandom00/logs", exist_ok=True)
    
    # Launch jobs for each sparsity configuration
    for neuron_sparsity, residual_sparsity in sparsity_configs:
        # Create a unique name for this configuration
        config_name = f"n{neuron_sparsity:.2f}_r{residual_sparsity:.2f}"
        
        # Create output directory
        output_dir = f"{SCRATCH}/iaifi_lab/Lab/ericjm/narrow/pruneandtrainrandom00/{config_name}"
        
        # Format the script with all parameters
        script = TEMPLATE.format(
            model_name=model_name,
            neuron_sparsity=neuron_sparsity,
            residual_sparsity=residual_sparsity,
            output_dir=output_dir,
            time_limit=training_params["time_limit"],
            max_length=training_params["max_length"],
            batch_size=training_params["batch_size"],
            accumulations=training_params["accumulations"],
            prune_samples=training_params["prune_samples"],
            train_skip=training_params["train_skip"],
            max_steps=training_params["max_steps"],
            lr=training_params["lr"],
            mask_steps=training_params["mask_steps"],
            eval_steps=training_params["eval_steps"],
            save_steps=training_params["save_steps"],
            limit_checkpoints=training_params["limit_checkpoints"],
            logging_steps=training_params["logging_steps"],
            warmup_steps=training_params["warmup_steps"],
        )
        
        # Write the script to a temporary file
        temp_script_path = f"/tmp/slurm_job_pruneandtrain_{config_name}.sh"
        with open(temp_script_path, "w") as f:
            f.write(script)
        
        # Make the script executable
        os.chmod(temp_script_path, 0o755)
        
        # Submit the job using sbatch
        print(f"Submitting job for configuration: neuron_sparsity={neuron_sparsity}, residual_sparsity={residual_sparsity}")
        result = subprocess.run(["sbatch", temp_script_path], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Job submitted successfully: {result.stdout.strip()}")
        else:
            print(f"Error submitting job: {result.stderr}")
        
        # Clean up temporary script file
        os.remove(temp_script_path)

if __name__ == "__main__":
    main()
