import os
import sys
from itertools import product
import subprocess

seeds = list(range(10))
widths = [128, 256, 512, 1024]
configs = list(product(seeds, widths))

if __name__ == '__main__':
    seed, width = configs[int(sys.argv[1])]
    subprocess.run([
        "python", 
        "/om2/user/ericjm/narrow/experiments/compositions0/train-nocurriculum.py",
        "--width", str(width),
        "--seed", str(seed),
        "--save-dir", f"/om/user/ericjm/results/narrow/compositions0/nocurriculum-width{width}-seed{seed}",
        "--wandb-project", "narrow-compositions0-nocurriculum"
    ])
