
import os
import sys
import copy
import pickle
from collections import defaultdict, OrderedDict

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn


def get_batch(n_tasks, n, Ss, codes, sizes, device=torch.device('cpu'), dtype=torch.float32):
    """Creates batch."""
    assert len(codes) == len(sizes)
    assert len(Ss) <= n_tasks
    assert 0 <= max([max(code) for code in codes]) < len(Ss)
    x = torch.zeros((sum(sizes), n_tasks+n), dtype=dtype, device=device)
    y = torch.zeros((sum(sizes),), dtype=torch.int64, device=device)
    rand_bits = torch.randint(low=0, high=2, size=(sum(sizes), n), device=device)
    x[:, n_tasks:] = rand_bits.float()
    idx = 0
    for code, size in zip(codes, sizes):
        if size > 0:
            # Union of subtasks
            S = sum([Ss[c] for c in code], [])
            assert len(S) == len(set(S))  # confirm disjointness
            x[idx:idx+size, code] = 1
            y[idx:idx+size] = torch.sum(x[idx:idx+size, n_tasks:][:, S], dim=1) % 2
            idx += size
    return x, y

def construct_mlp(layer_sizes, activation=nn.ReLU):
    """Constructs a multi-layer perceptron."""
    layers = OrderedDict()
    for i in range(len(layer_sizes) - 1):
        layers[f"fc{i}"] = nn.Linear(layer_sizes[i], layer_sizes[i+1])
        if i < len(layer_sizes) - 2:
            layers[f"act{i}"] = activation()
    return nn.Sequential(layers)

widths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] # 12
STEPS = 1_000_000

if __name__ == '__main__':
    widthi = int(sys.argv[1])
    width = widths[widthi]

    depth = 3
    activation_fn = nn.ReLU
    n_tasks = 2
    n = 64
    Ss = [
        [0, 1, 2],
        [3, 4, 5],
    ]
    codes = [
        [0, 1], # only train on compositional subtask
    ]
    train_sizes = [2_000] * len(codes)

    if __name__ == '__main__':

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        dtype = torch.float32
        torch.set_default_dtype(dtype)

        mlp = construct_mlp(
            [n_tasks + n] + [width] * (depth - 1) + [2],
            activation_fn
        ).to(dtype).to(device)

        ps = sum(p.numel() for p in mlp.parameters())
        print("Number of parameters:", ps)

        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3, eps=1e-5)
        loss_fn = nn.CrossEntropyLoss()

        steps = []
        losses = []
        subtask_losses = defaultdict(list)

        LOG_FREQ = 100

        for step in tqdm(range(STEPS)):
            # Perform a training step
            x, y = get_batch(n_tasks, n, Ss, codes, train_sizes, device=device, dtype=dtype)
            y_pred = mlp(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if step % LOG_FREQ == 0:
                steps.append(step)
                # Convert to Python float before storing
                losses.append(float(loss.item()))

                mlp.eval()
                with torch.no_grad():
                    for i, code in enumerate(codes):
                        # smaller batch for logging
                        x_log, y_log = get_batch(n_tasks, n, Ss, [code], [500], device=device, dtype=dtype)
                        y_log_pred = mlp(x_log)
                        log_loss = float(loss_fn(y_log_pred, y_log).item())
                        subtask_losses[i].append(log_loss)
                mlp.train()

        # Save final results with pickle
        results = {
            'steps': steps,
            'losses': losses,
            'subtask_losses': dict(subtask_losses),
            'codes': codes,
            'Ss': Ss,
            'train_sizes': train_sizes,
            'width': width,
        }
        with open(f'results-width{width:04d}.pkl', 'wb') as f:
            pickle.dump(results, f)

        # Save the model at the end of training
        # torch.save(mlp.state_dict(), 'model_final.pt')
