#!/usr/bin/env python
# coding: utf-8
"""
This script trains MLPs on multiple sparse parity problems at once,
including composite tasks.
"""

import argparse
from collections import defaultdict
import itertools
from typing import List, Tuple
import json
import pickle
import os

import torch
import torch.nn as nn
from tqdm.auto import tqdm

import wandb

def get_batch(
    n_tasks: int,
    n: int,
    subsets: List[List[int]],
    task_codes: List[List[int]],
    batch_sizes: List[int],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a batch for sparse parity tasks.

    Parameters
    ----------
    n_tasks : int
        Number of atomic tasks (must equal len(subsets)).
    n : int
        Length of each random bit-string.
    subsets : List[List[int]]
        Each subsets[i] is a list of zero-based bit-positions in [0..n-1].
    task_codes : List[List[int]]
        Which atomic tasks to combine for each sample.
    batch_sizes : List[int]
        Number of samples per code; same length as task_codes.
    device : str
        Torch device.
    dtype : torch.dtype
        Dtype for `x`. Output `y` is torch.int64.

    Returns
    -------
    x : torch.Tensor, shape (sum(batch_sizes), n_tasks + n)
    y : torch.Tensor, shape (sum(batch_sizes),)
    """
    assert len(subsets) == n_tasks, "Need exactly one subset per atomic task"
    assert len(task_codes) == len(batch_sizes)

    total = sum(batch_sizes)
    x = torch.zeros((total, n_tasks + n), dtype=dtype, device=device)
    bits = torch.randint(0, 2, (total, n), dtype=dtype, device=device)
    x[:, n_tasks:] = bits

    y = torch.empty((total,), dtype=torch.int64, device=device)

    idx = 0
    for code, size in zip(task_codes, batch_sizes):
        if size <= 0:
            continue
        # Union of bit‐positions
        S = set(itertools.chain.from_iterable(subsets[c] for c in code))
        x[idx:idx+size, code] = 1
        slice_bits = bits[idx:idx+size][:, sorted(S)]
        y[idx:idx+size] = slice_bits.sum(dim=1).remainder(2).to(torch.int64)
        idx += size

    return x, y


def main():
    p = argparse.ArgumentParser(
        description="Train MLPs on multiple sparse parity problems"
    )
    p.add_argument("--n", type=int, default=64, help="bit string length")
    p.add_argument("--width", type=int, default=512, help="hidden layer width of MLPs")
    p.add_argument("--depth", type=int, default=3, help="depth of MLPs")
    p.add_argument(
        "--activation",
        type=str,
        default="ReLU",
        choices=["ReLU", "Tanh", "Sigmoid"],
        help="activation function",
    )
    p.add_argument("--layernorm", action="store_true", help="use layer normalization")
    p.add_argument(
        "--samples-per-task", type=int, default=2000, help="number of samples per task"
    )
    p.add_argument(
        "--steps", type=int, default=200_000, help="number of training steps"
    )
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to use",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="data type",
    )
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument("--save-dir", type=str, help="directory to save results")
    p.add_argument("--verbose", action="store_true", help="print verbose output")
    p.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="wandb project name (omit to not log)",
    )
    p.add_argument(
        "--codes",
        type=str,
        default="[[0], [1], [2], [3], [0, 1, 2, 3]]",
        help="string representation of task codes to evaluate (must be valid Python list of lists)",
    )

    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.dtype == "float32":
        dtype = torch.float32
    else:
        dtype = torch.float64

    device = torch.device(args.device)

    if args.activation == "ReLU":
        activation_fn = nn.ReLU
    elif args.activation == "Tanh":
        activation_fn = nn.Tanh
    else:
        activation_fn = nn.Sigmoid

    n_tasks = 4
    assert args.n >= 16
    Ss = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]
    
    try:
        codes = eval(args.codes)
        if not isinstance(codes, list) or not all(isinstance(code, list) for code in codes):
            raise ValueError("Codes must be a list of lists")
    except Exception as e:
        raise ValueError(f"Invalid codes format: {e}. Must be a valid Python list of lists.")
    
    train_sizes = [args.samples_per_task] * len(codes)

    layers = []
    for i in range(args.depth):
        if i == 0:
            if args.layernorm:
                layers.append(nn.LayerNorm(n_tasks + args.n))
            layers.append(nn.Linear(n_tasks + args.n, args.width))
            layers.append(activation_fn())
        elif i == args.depth - 1:
            if args.layernorm:
                layers.append(nn.LayerNorm(args.width))
            layers.append(nn.Linear(args.width, 2))
        else:
            if args.layernorm:
                layers.append(nn.LayerNorm(args.width))
            layers.append(nn.Linear(args.width, args.width))
            layers.append(activation_fn())

    mlp = nn.Sequential(*layers).to(dtype).to(device)

    ps = sum(p.numel() for p in mlp.parameters())
    print("Number of parameters:", ps)
    print("Codes:", codes)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr, eps=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    steps = []
    samples = []
    losses = []
    subtask_losses = defaultdict(list)

    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    for step in tqdm(range(args.steps), disable=not args.verbose):
        with torch.no_grad():
            for i, code in enumerate(codes):
                x, y = get_batch(
                    n_tasks,
                    args.n,
                    Ss,
                    [code],
                    [train_sizes[i]],
                    device=device,
                    dtype=dtype,
                )
                y_pred = mlp(x)
                subtask_losses[i].append(loss_fn(y_pred, y).item())

        x, y = get_batch(
            n_tasks, args.n, Ss, codes, train_sizes, device=device, dtype=dtype
        )
        y_pred = mlp(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        steps.append(step)
        losses.append(loss.item())
        samples.append(x.shape[0])

        if use_wandb and step % 1000 == 0:
            wandb.log({"loss": loss.item()}, step=step)

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(mlp.state_dict(), os.path.join(args.save_dir, "model.pt"))

    results = {
        "steps": steps,
        "losses": losses,
        "subtask_losses": subtask_losses,
        "Ss": Ss,
        "codes": codes,
        "n_parameters": ps,
        "samples": samples,
    }

    with open(os.path.join(args.save_dir, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f)


if __name__ == "__main__":
    main()
