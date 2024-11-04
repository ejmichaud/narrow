import os
import sys
from itertools import product
import subprocess

seeds = list(range(5))
sample_sizes = [2000, 20000]
configs = list(product(seeds, sample_sizes))

if __name__ == '__main__':
    seed, samples = configs[int(sys.argv[1])]
    subprocess.run([
        "python", 
        "/om2/user/ericjm/narrow/experiments/compositions0/train-onlycomposite.py",
        "--width", str(512),
        "--seed", str(seed),
        "--samples-per-task", str(samples),
        "--save-dir", f"/om/user/ericjm/results/narrow/compositions0/onlycomposite-seed{seed}-samples{samples}",
        "--wandb-project", "narrow-compositions0-onlycomposite"
    ])
