#!/bin/bash

# Configuration
NUM_TO_KEEP=3  # Number of most recent checkpoints to keep in each directory

# Function to clean up checkpoints in a directory
cleanup_checkpoints() {
    local dir="$1"
    echo "Processing directory: $dir"
    
    # Get all checkpoint directories sorted by number
    checkpoints=($(find "$dir" -maxdepth 1 -type d -name "checkpoint-*" | sort -t- -k2 -n))
    
    # If there are no checkpoints, exit function
    if [ ${#checkpoints[@]} -eq 0 ]; then
        echo "No checkpoints found in $dir"
        return
    fi
    
    # Determine which checkpoints to keep
    declare -A keep_checkpoints
    
    # Keep multiples of 10k
    for checkpoint in "${checkpoints[@]}"; do
        checkpoint_num=$(basename "$checkpoint" | sed 's/checkpoint-//')
        if [ $((checkpoint_num % 10000)) -eq 0 ]; then
            keep_checkpoints["$checkpoint"]=1
            echo "Keeping $checkpoint (multiple of 10k)"
        fi
    done
    
    # Keep the last few checkpoints
    total_checkpoints=${#checkpoints[@]}
    start_idx=$((total_checkpoints - NUM_TO_KEEP))
    if [ $start_idx -lt 0 ]; then
        start_idx=0
    fi
    
    for ((i=start_idx; i<total_checkpoints; i++)); do
        keep_checkpoints["${checkpoints[$i]}"]=1
        echo "Keeping ${checkpoints[$i]} (among last $NUM_TO_KEEP)"
    done
    
    # Delete checkpoints that aren't in the keep list
    for checkpoint in "${checkpoints[@]}"; do
        if [ -z "${keep_checkpoints[$checkpoint]}" ]; then
            echo "Deleting $checkpoint"
            rm -rf "$checkpoint"
        fi
    done
    
    echo "Finished processing $dir"
    echo "-------------------------"
}

# Main script

# Get all lambda directories
lambda_dirs=($(find . -maxdepth 1 -type d -name "lambda_*"))

if [ ${#lambda_dirs[@]} -eq 0 ]; then
    echo "No lambda directories found"
    exit 1
fi

echo "Starting checkpoint cleanup..."
echo "Will keep multiples of 10,000 and the last $NUM_TO_KEEP checkpoints in each directory."
echo "-------------------------"

# Process each lambda directory
for lambda_dir in "${lambda_dirs[@]}"; do
    cleanup_checkpoints "$lambda_dir"
done

echo "Checkpoint cleanup complete!"
