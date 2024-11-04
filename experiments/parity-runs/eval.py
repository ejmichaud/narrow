import os
import sys
from itertools import product
import subprocess

widths = [8, 16, 32, 48, 64, 96, 192, 224, 288, 320, 384, 512]

if __name__ == '__main__':
    width = widths[int(sys.argv[1])]
    subprocess.run([
        "python", 
        "/om2/user/ericjm/narrow/scripts/sparse-parity-v6.py",
        "--n", "64",
        "--n-tasks", "32",
        "--k", "3",
        "--alpha", "0.3",
        "--width", str(width),
        "--steps", "1000000",
        "--save-dir", f"/om/user/ericjm/results/narrow/parity-runs/n64t32w{width}",
        "--wandb-project", "narrow"
    ])
