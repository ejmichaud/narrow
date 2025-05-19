#!/usr/bin/env python
# coding: utf-8
"""
This script trains MLPs on multiple sparse parity problems at once,
including composite tasks.
"""

from typing import Dict, List, Callable
import argparse
from collections import defaultdict
from itertools import islice
import copy
import json
import pickle
import os
import random

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

def get_batch(n_tasks, n, Ss, codes, sizes, device="cpu", dtype=torch.float32):
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
    assert len(Ss) <= n_tasks  # allow inequality ever?
    assert 0 <= max([max(code) for code in codes]) < len(Ss)  # codes are incides of Ss
    x = torch.zeros((sum(sizes), n_tasks + n), dtype=dtype, device=device)
    y = torch.zeros((sum(sizes),), dtype=torch.int64, device=device)
    x[:, n_tasks:] = torch.randint(
        low=0, high=2, size=(sum(sizes), n), dtype=dtype, device=device
    )
    idx = 0
    for code, size in zip(codes, sizes):
        if size > 0:
            S = sum([Ss[c] for c in code], [])  # union of subtasks
            assert len(S) == len(set(S))  # confirm disjointness
            x[idx : idx + size, code] = 1
            y[idx : idx + size] = (
                torch.sum(x[idx : idx + size, n_tasks:][:, S], dim=1) % 2
            )
            idx += size
    return x, y

def parse_args():
    parser = argparse.ArgumentParser(description='Train MLPs on sparse parity problems')
    parser.add_argument('--n', type=int, default=18, help='Bit string length for sparse parity problem')
    parser.add_argument('--width', type=int, default=64, help='Width of hidden layers')
    parser.add_argument('--depth', type=int, default=3, help='Depth of the network')
    parser.add_argument('--activation', type=str, default='ReLU', choices=['ReLU', 'Tanh', 'Sigmoid'], help='Activation function')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--layernorm', action='store_true', help='Use layer normalization')
    parser.add_argument('--samples_per_task', type=int, default=2000, help='Number of samples per task')
    parser.add_argument('--steps', type=int, default=20_000, help='Number of training steps')
    parser.add_argument('--verbose', action='store_true', help='Show progress bar')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float64'], help='Data type')
    return parser.parse_args()

args = parse_args()
Ss = [
    [0, 1, 2,],
    [3, 4, 5],
    [6, 7, 8],
    [9, 10, 11],
    [12, 13, 14],
    [15, 16, 17],
]
n_tasks = len(Ss)
codes = [[i] for i in range(n_tasks)] + [[0,1,2]] + [[3,4,5]]
train_sizes = [args.samples_per_task] * len(codes)
def train():

    random.seed(args.seed)
    np.random.seed(args.seed)
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
    samples = []
    losses = []
    subtask_losses = defaultdict(list)

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

    results = {
        "steps": steps,
        "losses": losses,
        "subtask_losses": subtask_losses,
        "Ss": Ss,
        "codes": codes,
        "n_parameters": ps,
        "samples": samples,
    }

    return mlp, results, device, dtype, loss_fn

mlp, results, device, dtype, loss_fn = train()
for i in range(len(codes)):
    print(f"code {codes[i]} loss: {results['subtask_losses'][i][-1]:.10f}")

def compute_ablation_scores(
        model: nn.Sequential, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        loss_fn: Callable) -> Dict[int, List[float]]:
    """
    Returns a list of ablation scores for each hidden neuron in the model.
    """
    scores = defaultdict(list)
    baseline_loss = loss_fn(model(x), y).item()
    for neuroni in range(model[0].out_features):
        mlp_ablation = copy.deepcopy(model)
        mlp_ablation[0].weight.data[neuroni, :] = 0
        mlp_ablation[0].bias.data[neuroni] = 0
        mlp_ablation[2].weight.data[:, neuroni] = 0
        y_pred = mlp_ablation(x)
        loss = loss_fn(y_pred, y)
        scores[0].append(abs(loss.item() - baseline_loss))
    for neuroni in range(model[2].out_features):
        mlp_ablation = copy.deepcopy(model)
        mlp_ablation[2].weight.data[neuroni, :] = 0
        mlp_ablation[2].bias.data[neuroni] = 0
        mlp_ablation[4].weight.data[:, neuroni] = 0
        y_pred = mlp_ablation(x)
        loss = loss_fn(y_pred, y)
        scores[2].append(abs(loss.item() - baseline_loss))
    return scores

ablation_scores = dict()
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
    ablation_scores[i] = compute_ablation_scores(mlp, x, y, loss_fn)


def prune_by_scores(
        model: nn.Sequential, 
        scores: Dict[int, List[float]], 
        sparsity: float) -> nn.Sequential:
    """
    Returns a copy of the model with the specified sparsity.
    Assumes that `model` alternates between linear layers and non-linearities.
    """
    retained = {i: torch.ones(len(neuron_scores), dtype=torch.bool) 
                for i, neuron_scores in scores.items()}
    n_neurons = sum(s.numel() for s in retained.values())
    layer_neuron_pairs = []
    for i, neuron_scores in scores.items():
        for ni in range(len(neuron_scores)):
            layer_neuron_pairs.append((i, ni, neuron_scores[ni]))
    layer_neuron_pairs.sort(key=lambda x: x[2])
    for layer, ni, _ in islice(layer_neuron_pairs, 0, int(n_neurons * sparsity)):
        retained[layer][ni] = False
    
    pruned = copy.deepcopy(model)
    layeris = sorted(scores.keys())
    for l in range(len(layeris)):
        pruned[layeris[l]].weight.data = model[layeris[l]].weight.data[retained[layeris[l]]]
        pruned[layeris[l]].bias.data = model[layeris[l]].bias.data[retained[layeris[l]]]
        if l == 0:
            pruned[layeris[l]].in_features = model[layeris[l]].in_features
            pruned[layeris[l]].out_features = sum(retained[layeris[l]])
            pruned[layeris[l]].weight.data = model[layeris[l]].weight.data[retained[layeris[l]], :]
            pruned[layeris[l]].bias.data = model[layeris[l]].bias.data[retained[layeris[l]]]
        else:
            pruned[layeris[l]].in_features = sum(retained[layeris[l-1]])
            pruned[layeris[l]].out_features = sum(retained[layeris[l]])
            pruned[layeris[l]].weight.data = model[layeris[l]].weight.data[retained[layeris[l]], :][:, retained[layeris[l-1]]]
            pruned[layeris[l]].bias.data = model[layeris[l]].bias.data[retained[layeris[l]]]
    # last layer doesn't have scores
    pruned[-1].in_features = sum(retained[layeris[l]])
    pruned[-1].out_features = model[-1].out_features
    pruned[-1].weight.data = model[-1].weight.data[:, retained[layeris[l]]]
    pruned[-1].bias.data = model[-1].bias.data
    return pruned

mlp_copy = copy.deepcopy(mlp)
mlp_copy
ablation_scores_all = ablation_scores
sparsities = np.arange(0.01, 0.99, 0.01)

losses_ablation = []
accuracies_ablation = []
losses_recovery_ablation = []
accuracies_recovery_ablation = []

losses_ablation_othertask = []
accuracies_ablation_othertask = []
losses_recovery_othertask = []
accuracies_recovery_othertask = []

subtaski = codes.index([0,1,2])
ablation_scores = ablation_scores_all[subtaski]
subtaskj = codes.index([3,4,5])

RECOVERY_STEPS = 1_000
for sparsity in tqdm(sparsities):
    pruned = prune_by_scores(mlp, ablation_scores, sparsity)
    x, y = get_batch(n_tasks, args.n, Ss, [codes[subtaski]], [5000], device=device, dtype=dtype)
    losses_ablation.append(loss_fn(pruned(x), y).item())
    accuracies_ablation.append(pruned(x).argmax(dim=1).eq(y).float().mean().item())
    optimizer = torch.optim.Adam(pruned.parameters(), lr=1e-3)
    for step in range(RECOVERY_STEPS):
        x, y = get_batch(n_tasks, args.n, Ss, [codes[subtaski]], [5000], device=device, dtype=dtype)
        y_pred = pruned(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses_recovery_ablation.append(loss_fn(pruned(x), y).item())
    accuracies_recovery_ablation.append(pruned(x).argmax(dim=1).eq(y).float().mean().item())

    pruned = prune_by_scores(mlp, ablation_scores, sparsity)
    x, y = get_batch(n_tasks, args.n, Ss, [codes[subtaskj]], [5000], device=device, dtype=dtype)
    losses_ablation_othertask.append(loss_fn(pruned(x), y).item())
    accuracies_ablation_othertask.append(pruned(x).argmax(dim=1).eq(y).float().mean().item())
    optimizer = torch.optim.Adam(pruned.parameters(), lr=1e-3)
    for step in range(RECOVERY_STEPS):
        x, y = get_batch(n_tasks, args.n, Ss, [codes[subtaskj]], [5000], device=device, dtype=dtype)
        y_pred = pruned(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses_recovery_othertask.append(loss_fn(pruned(x), y).item())
    accuracies_recovery_othertask.append(pruned(x).argmax(dim=1).eq(y).float().mean().item())
results_dir = f'results/width{args.width}/seed{args.seed}'
os.makedirs(results_dir, exist_ok=True)

# Save sparsities
with open(os.path.join(results_dir, 'sparsities.pkl'), 'wb') as f:
    pickle.dump(sparsities, f)

# Save ablation scores
with open(os.path.join(results_dir, 'ablation_scores.pkl'), 'wb') as f:
    pickle.dump(ablation_scores_all, f)

# Save model
torch.save(mlp.state_dict(), os.path.join(results_dir, 'model.pt'))

# Save training configuration
with open(os.path.join(results_dir, 'args.json'), 'w') as f:
    json.dump(vars(args), f, indent=2)

# Save training results
if 'results' in locals():
    with open(os.path.join(results_dir, 'training_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

# Save ablation and recovery statistics
ablation_results = {
    'losses_ablation': losses_ablation,
    'accuracies_ablation': accuracies_ablation,
    'losses_recovery_ablation': losses_recovery_ablation,
    'accuracies_recovery_ablation': accuracies_recovery_ablation,
    'losses_ablation_othertask': losses_ablation_othertask,
    'accuracies_ablation_othertask': accuracies_ablation_othertask,
    'losses_recovery_othertask': losses_recovery_othertask,
    'accuracies_recovery_othertask': accuracies_recovery_othertask
}

with open(os.path.join(results_dir, 'ablation_results.pkl'), 'wb') as f:
    pickle.dump(ablation_results, f)

# Save task information
task_info = {
    'Ss': Ss,
    'n_tasks': n_tasks,
    'codes': codes
}

with open(os.path.join(results_dir, 'task_info.pkl'), 'wb') as f:
    pickle.dump(task_info, f)

print(f"All results saved to {results_dir}")


# now let's do group lasso regularized training on [0,1,2]

# let's now just do regularization while preserving the first compositional subtask

mlp = copy.deepcopy(mlp_copy)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

def group_lasso_norm(model: nn.Sequential) -> torch.Tensor:
    # assumes model[0] is first linear, model[2] is second linear, and model[4] is third linear
    W1 = model[0].weight       # shape: (hidden, input)
    W2 = model[2].weight       # shape: (output, hidden)
    b1 = model[0].bias         # shape: (hidden,)
    W3 = model[4].weight
    b2 = model[2].bias
    layer0_norms = torch.sqrt(W1.pow(2).sum(dim=1) + b1.pow(2) + W2.pow(2).sum(dim=0))
    layer1_norms = torch.sqrt(W2.pow(2).sum(dim=1) + b2.pow(2) + W3.pow(2).sum(dim=0))
    return layer0_norms.sum() + layer1_norms.sum()

# Fine-tune on one atomic subtask with lasso regularization
subtaskis  = [codes.index([0, 1, 2])]                  # index of the chosen subtask with code [0, 1, 2]
norm_steps = 10_000
norm_lr    = 1e-3
lbda       = 1e-3
samples_per_batch = args.samples_per_task * len(subtaskis)

print("codes: ", [codes[si] for si in subtaskis])

# Reconfigure optimizer for fine-tuning
ft_optimizer = torch.optim.Adam(mlp.parameters(), lr=norm_lr)

# Tracking
lasso_vals      = []
prediction_vals = []
total_vals      = []

subtask_losses_regularization = defaultdict(list)

val_batches = []
for code in codes:
    x_val, y_val = get_batch(
        n_tasks, args.n, Ss, [code], [args.samples_per_task],
        device=device, dtype=dtype
    )
    val_batches.append((x_val, y_val))

for step in tqdm(range(norm_steps), desc='Lasso fine-tuning'):
    # single-subtask batch
    x, y = get_batch(
        n_tasks, args.n, Ss,
        [codes[si] for si in subtaskis], 
        [samples_per_batch // len(subtaskis)] * len(subtaskis),
        device=device, dtype=dtype
    )

    y_pred = mlp(x)
    pred_loss = loss_fn(y_pred, y)
    lasso_loss = group_lasso_norm(mlp)
    total_loss = pred_loss + lbda * lasso_loss

    ft_optimizer.zero_grad()
    total_loss.backward()
    ft_optimizer.step()

    prediction_vals.append(pred_loss.item())
    lasso_vals.append(lasso_loss.item())
    total_vals.append(total_loss.item())

    with torch.no_grad():
        for i, (x_val, y_val) in enumerate(val_batches):
            y_val_pred = mlp(x_val)
            subtask_losses_regularization[i].append(loss_fn(y_val_pred, y_val).item())

training_results_regularization = {
    'lasso_vals': lasso_vals,
    'prediction_vals': prediction_vals,
    'total_vals': total_vals,
    'subtask_losses_regularization': subtask_losses_regularization
}

# save regularized mlp and training results
torch.save(mlp.state_dict(), os.path.join(results_dir, 'model_regularized.pt'))

# save training results
with open(os.path.join(results_dir, 'training_results_regularization.pkl'), 'wb') as f:
    pickle.dump(training_results_regularization, f)

# now re-compute the ablation scores and sparsity curves for the regularized model

ablation_scores_regularization = dict()
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
    ablation_scores_regularization[i] = compute_ablation_scores(mlp, x, y, loss_fn)


ablation_scores_all = ablation_scores_regularization
sparsities = np.arange(0.01, 0.99, 0.01)

losses_ablation = []
accuracies_ablation = []
losses_recovery_ablation = []
accuracies_recovery_ablation = []

losses_ablation_othertask = []
accuracies_ablation_othertask = []
losses_recovery_othertask = []
accuracies_recovery_othertask = []

subtaski = codes.index([0,1,2])
ablation_scores = ablation_scores_all[subtaski]
subtaskj = codes.index([3,4,5])

RECOVERY_STEPS = 1_000
for sparsity in tqdm(sparsities):
    pruned = prune_by_scores(mlp, ablation_scores, sparsity)
    x, y = get_batch(n_tasks, args.n, Ss, [codes[subtaski]], [5000], device=device, dtype=dtype)
    losses_ablation.append(loss_fn(pruned(x), y).item())
    accuracies_ablation.append(pruned(x).argmax(dim=1).eq(y).float().mean().item())
    optimizer = torch.optim.Adam(pruned.parameters(), lr=1e-3)
    for step in range(RECOVERY_STEPS):
        x, y = get_batch(n_tasks, args.n, Ss, [codes[subtaski]], [5000], device=device, dtype=dtype)
        y_pred = pruned(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses_recovery_ablation.append(loss_fn(pruned(x), y).item())
    accuracies_recovery_ablation.append(pruned(x).argmax(dim=1).eq(y).float().mean().item())

    pruned = prune_by_scores(mlp, ablation_scores, sparsity)
    x, y = get_batch(n_tasks, args.n, Ss, [codes[subtaskj]], [5000], device=device, dtype=dtype)
    losses_ablation_othertask.append(loss_fn(pruned(x), y).item())
    accuracies_ablation_othertask.append(pruned(x).argmax(dim=1).eq(y).float().mean().item())
    optimizer = torch.optim.Adam(pruned.parameters(), lr=1e-3)
    for step in range(RECOVERY_STEPS):
        x, y = get_batch(n_tasks, args.n, Ss, [codes[subtaskj]], [5000], device=device, dtype=dtype)
        y_pred = pruned(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses_recovery_othertask.append(loss_fn(pruned(x), y).item())
    accuracies_recovery_othertask.append(pruned(x).argmax(dim=1).eq(y).float().mean().item())

ablation_results_regularization = {
    'losses_ablation': losses_ablation,
    'accuracies_ablation': accuracies_ablation,
    'losses_recovery_ablation': losses_recovery_ablation,
    'accuracies_recovery_ablation': accuracies_recovery_ablation,
    'losses_ablation_othertask': losses_ablation_othertask,
    'accuracies_ablation_othertask': accuracies_ablation_othertask,
    'losses_recovery_othertask': losses_recovery_othertask,
    'accuracies_recovery_othertask': accuracies_recovery_othertask
}

with open(os.path.join(results_dir, 'ablation_results_regularization.pkl'), 'wb') as f:
    pickle.dump(ablation_results_regularization, f)

