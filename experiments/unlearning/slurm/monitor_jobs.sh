#!/bin/bash
# Quick job monitoring script

echo "=== Active Jobs ==="
squeue -u $USER -o "%.10i %.40j %.8T %.10M %.10l"

echo ""
echo "=== Recent Progress (last 5 lines of each running job) ==="
for log in slurm_logs/python_train_*_427920*.out; do
    if [ -f "$log" ]; then
        jobname=$(basename $log .out | sed 's/python_train_//' | sed 's/_[0-9]*$//')
        echo ""
        echo "--- $jobname ---"
        tail -5 $log 2>/dev/null | grep -E "Training|step|%|loss" || echo "No training output yet"
    fi
done
