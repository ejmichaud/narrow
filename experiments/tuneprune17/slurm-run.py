#!/usr/bin/env python3
import os
import subprocess

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=tuneprune17
#SBATCH --partition=gpu
#SBATCH --output=/n/home04/ericjm/narrow/experiments/tuneprune17/logs/slurm-%A_%a.out
#SBATCH --time={time_limit}
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=20GB

python $HOME/narrow/tuneprune-autoadjust.py \
    --model_name {model_name} \
    --output_dir {output_dir} \
    --sparsity_lambda {sparsity_lambda} \
    --regularizer {regularizer} \
    --lr {lr} \
    --max_steps {max_steps} \
    --limit_checkpoints {limit_checkpoints} \
    --save_steps {save_steps} \
    --batch_size {batch_size} \
    --accumulations {accumulations} \
    --max_length {max_length} \
    --logging_steps {logging_steps} \
    --use_streaming \
"""

def main():
    # Define the base model to use
    model_name = "NousResearch/Llama-3.2-1B"
    
    # Define the configurations
    # 3 configurations for Hoyer loss on neurons
    # 3 configurations for Hoyer loss on residual stream
    configs = [
        {
            "name": "hoyer_neurons_0.0001",
            "regularizer": "group_hoyer_neurons",
            "sparsity_lambda": "0.0001",
        },
        {
            "name": "hoyer_neurons_0.005",
            "regularizer": "group_hoyer_neurons",
            "sparsity_lambda": "0.005",
        },
        {
            "name": "hoyer_neurons_0.1",
            "regularizer": "group_hoyer_neurons",
            "sparsity_lambda": "0.1",
        },
        {
            "name": "hoyer_residual_0.001",
            "regularizer": "group_hoyer_residual_stream",
            "sparsity_lambda": "0.001",
        },
        {
            "name": "hoyer_residual_0.05",
            "regularizer": "group_hoyer_residual_stream",
            "sparsity_lambda": "0.05",
        },
        {
            "name": "hoyer_residual_0.5",
            "regularizer": "group_hoyer_residual_stream",
            "sparsity_lambda": "0.5",
        }
    ]

    # Common training parameters
    training_params = {
        "time_limit": "1-00:00:00",
        "lr": "1e-5",
        "max_steps": "20000",
        "batch_size": 8,
        "accumulations": 8,
        "save_steps": "5000",
        "limit_checkpoints": "-1",
        "max_length": 1024,
        "logging_steps": 5,
    }

    # Launch each job sequentially
    SCRATCH = os.environ.get('SCRATCH')
    for config in configs:
        # Create output directory
        output_dir = f"{SCRATCH}/iaifi_lab/Lab/ericjm/narrow/tuneprune17/{config['name']}"
        
        # Format the script with all parameters
        script = TEMPLATE.format(
            model_name=model_name,
            output_dir=output_dir,
            sparsity_lambda=config["sparsity_lambda"],
            regularizer=config["regularizer"],
            time_limit=training_params["time_limit"],
            lr=training_params["lr"],
            max_steps=training_params["max_steps"],
            batch_size=training_params["batch_size"],
            accumulations=training_params["accumulations"],
            save_steps=training_params["save_steps"],
            limit_checkpoints=training_params["limit_checkpoints"],
            max_length=training_params["max_length"],
            logging_steps=training_params["logging_steps"],
        )
        
        # Write the script to a temporary file
        temp_script_path = f"/tmp/slurm_job_tuneprune17_{config['name']}.sh"
        with open(temp_script_path, "w") as f:
            f.write(script)
        
        # Make the script executable
        os.chmod(temp_script_path, 0o755)
        
        # Submit the job using sbatch
        print(f"Submitting job for configuration: {config['name']}")
        result = subprocess.run(["sbatch", temp_script_path], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Job submitted successfully: {result.stdout.strip()}")
        else:
            print(f"Error submitting job: {result.stderr}")
        
        # Optional: remove the temporary file after submission
        # os.remove(temp_script_path)

if __name__ == "__main__":
    main() 