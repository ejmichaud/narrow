import os
import sys
from itertools import product
import subprocess

widths = [96, 192, 256, 320, 512, 1024, 2048]
l1s = [0.01, 0.1, 0.5, 1.0]

configs = list(product(widths, l1s)) # 28

if __name__ == '__main__':
    width, l1 = configs[int(sys.argv[1])]
    subprocess.run([
        "python", 
        "/om2/user/ericjm/narrow/scripts/sparse-parity-v6-l1.py",
        "--n", "64",
        "--n-tasks", "32",
        "--k", "3",
        "--alpha", "0.3",
        "--width", str(width),
        "--l1", str(l1),
        "--steps", "200000",
        "--save-dir", f"/om/user/ericjm/results/narrow/parity-runs-l1/n64t32w{width}l{l1}",
        "--wandb-project", "narrow-parity-l1"
    ])
