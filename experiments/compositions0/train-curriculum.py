#!/usr/bin/env python
# coding: utf-8
"""
This script trains MLPs on multiple sparse parity problems at once,
including composite tasks.
"""

import argparse
from collections import defaultdict
import json
import pickle
import os

import torch
import torch.nn as nn
from tqdm.auto import tqdm

import wandb

def get_batch(n_tasks, n, Ss, codes, sizes, device='cpu', dtype=torch.float32):
    """Creates batch.

    Parameters
    ----------
    n_tasks : int
        Number of atomic subtasks.
    n : int
        Bit string length for sparse parity problem.
    Ss : list of lists of ints
        Subsets of [1, ... n] to compute sparse parities on.
    codes : list of list of int
        The subtask indices which the batch will consist of.
        When a sample is an atomic subtask i, the corresponding
        code is [i]. When the sample is a composite subtask
        consisting of subtasks i and j, the corresponding code
        is [i, j].
    sizes : list of int
        Number of samples for each task. Must be the same length
        as codes.
    device : str
        Device to put batch on.
    dtype : torch.dtype
        Data type to use for input x. Output y is torch.int64.

    Returns
    -------
    x : torch.Tensor
        inputs
    y : torch.Tensor
        labels
    """
    assert len(codes) == len(sizes)
    assert len(Ss) <= n_tasks # allow inequality ever?
    assert 0 <= max([max(code) for code in codes]) < len(Ss) # codes are incides of Ss
    x = torch.zeros((sum(sizes), n_tasks+n), dtype=dtype, device=device)
    y = torch.zeros((sum(sizes),), dtype=torch.int64, device=device)
    x[:, n_tasks:] = torch.randint(low=0, high=2, size=(sum(sizes), n), dtype=dtype, device=device)
    idx = 0
    for code, size in zip(codes, sizes):
        if size > 0:
            S = sum([Ss[c] for c in code], []) # union of subtasks
            assert len(S) == len(set(S)) # confirm disjointness
            x[idx:idx+size, code] = 1
            y[idx:idx+size] = torch.sum(x[idx:idx+size, n_tasks:][:, S], dim=1) % 2
            idx += size
    return x, y

def main():
    p = argparse.ArgumentParser(description='Train MLPs on multiple sparse parity problems')
    p.add_argument('--n', type=int, default=64, help='bit string length')
    p.add_argument('--width', type=int, default=512, help='hidden layer width of MLPs')
    p.add_argument('--depth', type=int, default=3, help='depth of MLPs')
    p.add_argument('--activation', type=str, default='ReLU', choices=['ReLU', 'Tanh', 'Sigmoid'], help='activation function')
    p.add_argument('--layernorm', action='store_true', help='use layer normalization')
    p.add_argument('--samples-per-task', type=int, default=2000, help='number of samples per task')
    p.add_argument('--steps', type=int, default=200_000, help='number of training steps')
    p.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device to use')
    p.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float64'], help='data type')
    p.add_argument('--seed', type=int, default=0, help='random seed')
    p.add_argument('--save-dir', type=str, help='directory to save results')
    p.add_argument('--verbose', action='store_true', help='print verbose output')
    p.add_argument('--wandb-project', type=str, default=None, help='wandb project name (omit to not log)')

    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.dtype == 'float32':
        dtype = torch.float32
    else:
        dtype = torch.float64

    device = torch.device(args.device)

    if args.activation == 'ReLU':
        activation_fn = nn.ReLU
    elif args.activation == 'Tanh':
        activation_fn = nn.Tanh
    else:
        activation_fn = nn.Sigmoid

    n_tasks = 8
    assert args.n >= 32
    Ss = [
        [0, 1, 2, 3], 
        [4, 5, 6, 7], 
        [8, 9, 10, 11], 
        [12, 13, 14, 15], 
        [16, 17, 18, 19], 
        [20, 21, 22, 23], 
        [24, 25, 26, 27], 
        [28, 29, 30, 31]
    ]
    codes = [[0], [1], [2], [3], [0, 1], [2, 3], [0, 1, 2, 3]]
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
    # print("Number of parameters:", ps)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr, eps=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    steps = []
    losses = []
    subtask_losses = defaultdict(list)

    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    for step in tqdm(range(args.steps), disable=not args.verbose):
        with torch.no_grad():
            for i, code in enumerate(codes):
                x, y = get_batch(n_tasks, args.n, Ss, [code], [train_sizes[i]], device=device, dtype=dtype)
                y_pred = mlp(x)
                subtask_losses[i].append(loss_fn(y_pred, y).item())

        x, y = get_batch(n_tasks, args.n, Ss, codes, train_sizes, device=device, dtype=dtype)
        y_pred = mlp(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        steps.append(step)
        losses.append(loss.item())

        if use_wandb and step % 1000 == 0:
            wandb.log({'loss': loss.item()}, step=step)

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(mlp.state_dict(), os.path.join(args.save_dir, "model.pt"))

    results = {
        'steps': steps,
        'losses': losses,
        'subtask_losses': subtask_losses,
        'Ss': Ss,
        'codes': codes,
        'n_parameters': ps,
    }

    with open(os.path.join(args.save_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f)

if __name__ == '__main__':
    main()
