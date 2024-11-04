# import os
# import argparse
import copy
from collections import defaultdict, OrderedDict
# from itertools import islice, product
# import random
# import json
import pickle

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn

from nnsight import NNsight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
dtype = torch.float32
torch.set_default_dtype(dtype)

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


def construct_mlp(layer_sizes, activation=nn.ReLU):
    """Constructs a multi-layer perceptron with variable layer widths.
    Parameters
    ----------
    layer_sizes : list of int
        List of integers specifying the width of each layer, including input and output layers.
        For example, [64, 128, 256, 10] would create a network with:
        - 64 input neurons
        - 128 neurons in the first hidden layer
        - 256 neurons in the second hidden layer
        - 10 output neurons
    activation : torch.nn.Module
        Activation function to use between layers (default: nn.ReLU).
    Returns
    -------
    mlp : torch.nn.Sequential
        Multi-layer perceptron.
    """
    layers = OrderedDict()
    for i in range(len(layer_sizes) - 1):
        layers[f"fc{i}"] = nn.Linear(layer_sizes[i], layer_sizes[i+1])
        if i < len(layer_sizes) - 2:  # Don't add activation after the last layer
            layers[f"act{i}"] = activation()
    return nn.Sequential(layers)


depth = 3
width = 2048
activation_fn = nn.ReLU

n_tasks = 24 # number of atomic subtasks
n = 96       # 24 * 4
Ss = [[i, i+1, i+2, i+3] for i in range(0, 24*4, 4)]
codes = [
    [0], [1], [2], [3],
    [0, 1], [2, 3],
    [0, 1, 2, 3],
    [4], [5], [6], [7],
    [4, 5], [6, 7],
    [4, 5, 6, 7],
    [8], [9], [10], [11],
    [8, 9], [10, 11],
    [8, 9, 10, 11],
    [12], [13], [14], [15],
    [12, 13], [14, 15],
    [12, 13, 14, 15],
    [16], [17], [18], [19],
    [16, 17], [18, 19],
    [16, 17, 18, 19],
    [20], [21], [22], [23],
    [20, 21], [22, 23],
    [20, 21, 22, 23]
]
train_sizes = [2_000] * len(codes)

mlp = construct_mlp(
    [n_tasks + n] + [width] * (depth - 1) + [2], 
    activation_fn
).to(dtype).to(device)

# compute total number of parameters in MLP
ps = 0
for p in mlp.parameters():
    ps += p.numel()
print("Number of parameters:", ps)

optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3, eps=1e-5)
loss_fn = nn.CrossEntropyLoss()

steps = []
losses = []
subtask_losses = defaultdict(list)
for step in tqdm(range(30_000)):
    with torch.no_grad():
        for i, code in enumerate(codes):
            x, y = get_batch(n_tasks, n, Ss, [code], [train_sizes[i]], device=device, dtype=dtype)
            y_pred = mlp(x)
            subtask_losses[i].append(loss_fn(y_pred, y).item())
    x, y = get_batch(n_tasks, n, Ss, codes, train_sizes, device=device, dtype=dtype)
    y_pred = mlp(x)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    steps.append(step)
    losses.append(loss.item())

# save pretraining metrics
with open("pretrain_metrics.pkl", "wb") as f:
    pickle.dump({
        "steps": steps,
        "losses": losses,
        "subtask_losses": subtask_losses,
    }, f)

# save pretraining model (cpu)
torch.save(mlp.cpu().state_dict(), "pretrain_model.pth")
