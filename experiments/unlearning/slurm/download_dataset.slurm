#!/bin/bash
#SBATCH --job-name=dl_dataset
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH -t 120
#SBATCH --output=slurm_logs/download_dataset_%j.out
#SBATCH --error=slurm_logs/download_dataset_%j.err

echo "================================"
echo "Downloading github-code Dataset"
echo "================================"
echo ""

cd ..

/n/home04/ericjm/.conda/envs/narrow/bin/python -c "
import os
os.environ['HF_HOME'] = os.environ.get('SCRATCH', '/tmp') + '/iaifi_lab/Lab/ericjm/.cache/huggingface'

from datasets import load_dataset

print('Downloading github-code dataset (Python language)')
print('Using XET storage for faster download')
print('Cache location:', os.environ['HF_HOME'])
print('')

dataset = load_dataset(
    'codeparrot/github-code',
    streaming=False,
    languages=['Python'],
    split='train',
    trust_remote_code=True
)

print('')
print('================================')
print('Dataset cached successfully!')
print('================================')
print(f'Total samples: {len(dataset):,}')
print(f'Cache location: {os.environ[\"HF_HOME\"]}')
"

echo ""
echo "Job completed at $(date)"

