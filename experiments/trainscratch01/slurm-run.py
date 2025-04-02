#!/usr/bin/env python3
import os
import subprocess

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=train_scratch
#SBATCH --account=iaifi_lab
#SBATCH --partition=iaifi_gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --output=/n/home04/ericjm/narrow/experiments/trainscratch01/logs/slurm-%A_%a.out
#SBATCH --time={time_limit}
#SBATCH --ntasks=1
#SBATCH --mem=12GB

python $HOME/narrow/trainscratch.py \
    --tokenizer_name "NousResearch/Meta-Llama-3.1-8B" \
    --hidden_size {hidden_size} \
    --num_layers {num_layers} \
    --num_heads {num_heads} \
    --intermediate_size {intermediate_size} \
    --output_dir {output_dir} \
    --lr {lr} \
    --max_steps {max_steps} \
    --max_length {max_length} \
    --batch_size {batch_size} \
    --accumulations {accumulations} \
    --logging_steps {logging_steps} \
    --limit_checkpoints {limit_checkpoints} \
    --use_streaming
"""

def main():
    # Define model configurations along with their desired time limits.
    model_configs = [
        # {"hidden_size": 256, "num_layers": 4, "num_heads": 4, "intermediate_size": 1024, "time_limit": "1-00:00:00"},
        # {"hidden_size": 512, "num_layers": 8, "num_heads": 8, "intermediate_size": 2048, "time_limit": "1-00:00:00"},
        # {"hidden_size": 768, "num_layers": 12, "num_heads": 12, "intermediate_size": 3072, "time_limit": "2-00:00:00"},
        # {"hidden_size": 864, "num_layers": 16, "num_heads": 16, "intermediate_size": 3456, "time_limit": "2-00:00:00"},
        # {"hidden_size": 1024, "num_layers": 16, "num_heads": 16, "intermediate_size": 4096, "time_limit": "3-00:00:00"},
        # {"hidden_size": 1280, "num_layers": 20, "num_heads": 20, "intermediate_size": 5120, "time_limit": "3-00:00:00"},
        {"hidden_size": 1536, "num_layers": 24, "num_heads": 24, "intermediate_size": 6144, "time_limit": "3-00:00:00", "batch_size": 4, "accumulations": 16},
        {"hidden_size": 1792, "num_layers": 28, "num_heads": 28, "intermediate_size": 7168, "time_limit": "3-00:00:00", "batch_size": 4, "accumulations": 16},
        {"hidden_size": 2048, "num_layers": 32, "num_heads": 32, "intermediate_size": 8192, "time_limit": "3-00:00:00", "batch_size": 4, "accumulations": 16},
    ]

    # Launch each job sequentially. Depending on your use case you might run them concurrently.
    for config in model_configs:
        # Create a unique output directory based on model size
        SCRATCH = os.environ.get('SCRATCH')
        output_dir = f"{SCRATCH}/iaifi_lab/Lab/ericjm/narrow/trainscratch01/d{config['hidden_size']}_l{config['num_layers']}_h{config['num_heads']}"
        script = TEMPLATE.format(
            time_limit=config["time_limit"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            intermediate_size=config["intermediate_size"],
            output_dir=output_dir,
            lr="5e-4",
            max_steps="100000",
            max_length="1024",
            batch_size=config["batch_size"],
            accumulations=config["accumulations"],
            logging_steps="5",
            save_steps="5000",
            limit_checkpoints="3",
        )
        
        # Write the script to a temporary file
        temp_script_path = f"/tmp/slurm_job_{config['hidden_size']}_{config['num_layers']}.sh"
        with open(temp_script_path, "w") as f:
            f.write(script)
        
        # Make the script executable
        os.chmod(temp_script_path, 0o755)
        
        # Submit the job using sbatch
        print(f"Submitting job for model with hidden_size={config['hidden_size']}, num_layers={config['num_layers']}")
        result = subprocess.run(["sbatch", temp_script_path], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Job submitted successfully: {result.stdout.strip()}")
        else:
            print(f"Error submitting job: {result.stderr}")
        
        # Optional: remove the temporary file after submission
        # os.remove(temp_script_path)

if __name__ == "__main__":
    main()
