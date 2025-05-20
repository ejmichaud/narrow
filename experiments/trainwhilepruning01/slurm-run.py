#!/usr/bin/env python3
import os
import subprocess

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=train_prune
#SBATCH --partition=iaifi_gpu
#SBATCH --account=iaifi_lab
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --output=/n/home04/ericjm/narrow/experiments/trainwhilepruning01/logs/slurm-%A_%a.out
#SBATCH --time={time_limit}
#SBATCH --ntasks=1
#SBATCH --mem=12GB

python $HOME/narrow/train_while_pruning.py \
    --model_name {model_name} \
    --neuron_sparsity {neuron_sparsity} \
    --neuron_prune_steps {neuron_prune_steps} \
    --neuron_prune_number {neuron_prune_number} \
    --residual_sparsity {residual_sparsity} \
    --residual_prune_steps {residual_prune_steps} \
    --residual_prune_number {residual_prune_number} \
    --output_dir {output_dir} \
    --lr {lr} \
    --max_steps {max_steps} \
    --max_length {max_length} \
    --batch_size {batch_size} \
    --accumulations {accumulations} \
    --logging_steps {logging_steps} \
    --save_steps {save_steps} \
    --limit_checkpoints {limit_checkpoints} \
    --mask_steps {mask_steps} \
    --ema_alpha {ema_alpha} \
    --pruning_warmup_steps {pruning_warmup_steps} \
    --use_streaming
"""

def main():
    # Define the base model to use
    model_name = "NousResearch/Llama-3.2-1B"
    
    # Define the three pruning configurations
    # 1. Neuron pruning only
    # 2. Residual stream pruning only
    # 3. Both neuron and residual stream pruning
    pruning_configs = [
        {
            "name": "neuron_only",
            "neuron_sparsity": 0.9,
            "neuron_prune_steps": 8,
            "neuron_prune_number": 32,
            "residual_sparsity": 0.0,
            "residual_prune_steps": -1,
            "residual_prune_number": 0,
        },
        {
            "name": "residual_only",
            "neuron_sparsity": 0.0,
            "neuron_prune_steps": -1,
            "neuron_prune_number": 0,
            "residual_sparsity": 0.8,
            "residual_prune_steps": 20,
            "residual_prune_number": 1,
        },
        {
            "name": "both",
            "neuron_sparsity": 0.9,
            "neuron_prune_steps": 8,
            "neuron_prune_number": 32,
            "residual_sparsity": 0.8,
            "residual_prune_steps": 20,
            "residual_prune_number": 1,
        }
    ]

    # Common training parameters
    training_params = {
        "time_limit": "1-00:00:00",
        "lr": "5e-4",
        "max_steps": "30000",
        "max_length": "1024",
        "batch_size": 8,
        "accumulations": 4,
        "logging_steps": "5",
        "save_steps": "5000",
        "limit_checkpoints": "-1",
        "mask_steps": "1",
        "ema_alpha": "0.99",
        "pruning_warmup_steps": "100",
    }

    # Launch each job sequentially
    SCRATCH = os.environ.get('SCRATCH')
    
    for config in pruning_configs:
        # Create a unique output directory based on pruning configuration
        output_dir = f"{SCRATCH}/iaifi_lab/Lab/ericjm/narrow/trainwhilepruning01/{config['name']}"
        
        script = TEMPLATE.format(
            model_name=model_name,
            neuron_sparsity=config["neuron_sparsity"],
            neuron_prune_steps=config["neuron_prune_steps"],
            neuron_prune_number=config["neuron_prune_number"],
            residual_sparsity=config["residual_sparsity"],
            residual_prune_steps=config["residual_prune_steps"],
            residual_prune_number=config["residual_prune_number"],
            output_dir=output_dir,
            time_limit=training_params["time_limit"],
            lr=training_params["lr"],
            max_steps=training_params["max_steps"],
            max_length=training_params["max_length"],
            batch_size=training_params["batch_size"],
            accumulations=training_params["accumulations"],
            logging_steps=training_params["logging_steps"],
            save_steps=training_params["save_steps"],
            limit_checkpoints=training_params["limit_checkpoints"],
            mask_steps=training_params["mask_steps"],
            ema_alpha=training_params["ema_alpha"],
            pruning_warmup_steps=training_params["pruning_warmup_steps"],
        )
        
        # Write the script to a temporary file
        temp_script_path = f"/tmp/slurm_job_trainwhilepruning_{config['name']}.sh"
        with open(temp_script_path, "w") as f:
            f.write(script)
        
        # Make the script executable
        os.chmod(temp_script_path, 0o755)
        
        # Submit the job using sbatch
        print(f"Submitting job for pruning configuration: {config['name']}")
        result = subprocess.run(["sbatch", temp_script_path], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Job submitted successfully: {result.stdout.strip()}")
        else:
            print(f"Error submitting job: {result.stderr}")
        
        # Optional: remove the temporary file after submission
        # os.remove(temp_script_path)

if __name__ == "__main__":
    main()