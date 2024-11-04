#!/usr/bin/env python
# coding: utf-8
"""
This script trains MLPs on multiple sparse parity problems at once.

Imposes an L1 penalty on the hidden layer activations to encourage sparsity.
"""

import os
import argparse
from collections import defaultdict
from itertools import islice, product
import random
import json
import pickle
import time

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn

import wandb

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len, device=self.tensors[0].device)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i:self.i+self.batch_size]
            batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)
        else:
            batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def get_batch(n_tasks, n, Ss, codes, sizes, device='cpu', dtype=torch.float32):
    """Creates batch. 

    Parameters
    ----------
    n_tasks : int
        Number of tasks.
    n : int
        Bit string length for sparse parity problem.
    Ss : list of lists of ints
        Subsets of [1, ... n] to compute sparse parities on.
    codes : list of int
        The subtask indices which the batch will consist of
    sizes : list of int
        Number of samples for each subtask
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
    batch_x = torch.zeros((sum(sizes), n_tasks+n), dtype=dtype, device=device)
    batch_y = torch.zeros((sum(sizes),), dtype=torch.int64, device=device)
    start_i = 0
    for (S, size, code) in zip(Ss, sizes, codes):
        if size > 0:
            x = torch.randint(low=0, high=2, size=(size, n), dtype=dtype, device=device)
            y = torch.sum(x[:, S], dim=1) % 2
            x_task_code = torch.zeros((size, n_tasks), dtype=dtype, device=device)
            x_task_code[:, code] = 1
            x = torch.cat([x_task_code, x], dim=1)
            batch_x[start_i:start_i+size, :] = x
            batch_y[start_i:start_i+size] = y
            start_i += size
    return batch_x, batch_y
    
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def main():
    p = argparse.ArgumentParser(description='Train MLPs on multiple sparse parity problems')
    p.add_argument('--n-tasks', type=int, default=100,
                   help='number of tasks')
    p.add_argument('--n', type=int, default=50,
                   help='bit string length')
    p.add_argument('--k', type=int, default=3,
                   help='sparsity of parity problems')
    p.add_argument('--alpha', type=float, default=1.5,
                   help='power law exponent for subtask sampling')
    p.add_argument('--offset', type=int, default=0,
                   help='offset for subtask coding')
    p.add_argument('--D', type=int, default=-1,
                   help='number of data points, -1 for infinite')
    p.add_argument('--width', type=int, default=100,
                   help='hidden layer width of MLPs')
    p.add_argument('--depth', type=int, default=2,
                   help='hidden layer depth of MLPs')
    p.add_argument('--activation', type=str, default='ReLU',
                   choices=['ReLU', 'Tanh', 'Sigmoid'],
                   help='activation function')
    p.add_argument('--steps', type=int, default=25000,
                   help='number of training steps')
    p.add_argument('--batch-size', type=int, default=10000,
                   help='training batch size')
    p.add_argument('--lr', type=float, default=1e-3,
                   help='learning rate')
    p.add_argument('--weight-decay', type=float, default=0.0,
                   help='weight decay')
    p.add_argument('--l1', type=float, default=0.0,
                   help='sparsity penalty coefficient')
    p.add_argument('--test-points', type=int, default=30000,
                   help='number of points to use for evaluation')
    p.add_argument('--test-points-per-task', type=int, default=1000,
                   help='number of points to use for evaluation per task')
    p.add_argument('--stop-early', action='store_true',
                   help='whether to use early stopping')
    p.add_argument('--device', type=str, default=None,
                   help='device to use')
    p.add_argument('--dtype', type=str, default='float32',
                   choices=['bfloat16', 'float16', 'float32', 'float64'],  
                   help='data type')
    p.add_argument('--log-freq', type=int, default=None,
                   help='log frequency (default log_freq=steps//1000)')
    p.add_argument('--seed', type=int, default=0,
                   help='random seed')
    p.add_argument("--save-dir", type=str,
                   default=None,
                   help="if None: '.' or /om/user/ericjm/results/{wandb_run_id} if using wandb")
    p.add_argument('--wandb-project', type=str, default=None,
                   help='wandb project name (omit to not log)')
    p.add_argument('--verbose', action='store_true',
                   help='whether to print progress updates')

    args = p.parse_args()

    assert args.depth == 2, "Only depth 2 is supported (one hidden layer)"

    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    supported_types = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
    }
    if args.dtype not in supported_types:
        raise ValueError(f'Invalid dtype: {args.dtype}')
    else:
        dtype = supported_types[args.dtype]
    if args.log_freq is None:
        args.log_freq = max(1, args.steps // 1000) # overwrites None
        
    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project, 
            config=vars(args),
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_default_dtype(dtype)

    if args.activation == 'ReLU':
        activation_fn = nn.ReLU
    elif args.activation == 'Tanh':
        activation_fn = nn.Tanh
    elif args.activation == 'Sigmoid':
        activation_fn = nn.Sigmoid
    else:
        raise ValueError(f'Invalid activation function: {args.activation}')

    # create model
    layers = []
    for i in range(args.depth):
        if i == 0:
            layers.append(nn.Linear(args.n_tasks + args.n, args.width))
            layers.append(activation_fn())
        elif i == args.depth - 1:
            layers.append(nn.Linear(args.width, 2))
        else:
            layers.append(nn.Linear(args.width, args.width))  
            layers.append(activation_fn())
    mlp = nn.Sequential(*layers).to(dtype).to(device)

    Ss = []
    for _ in range(args.n_tasks * 10):
        S = tuple(sorted(list(random.sample(range(args.n), args.k))))
        if S not in Ss:
            Ss.append(S)
        if len(Ss) == args.n_tasks:
            break
    assert len(Ss) == args.n_tasks, "Couldn't find enough subsets for tasks for the given n, k"

    probs = np.array([n ** (-args.alpha) for n in range(1+args.offset, args.n_tasks+args.offset+1)])
    probs = probs / np.sum(probs)
    cdf = np.cumsum(probs)

    test_batch_sizes = [int(prob * args.test_points) for prob in probs]
    
    if args.D != -1:
        samples = np.searchsorted(cdf, np.random.rand(args.D,))
        hist, _ = np.histogram(samples, bins=args.n_tasks, range=(0, args.n_tasks-1))
        train_x, train_y = get_batch(n_tasks=args.n_tasks, n=args.n, Ss=Ss, codes=list(range(args.n_tasks)), sizes=hist, device='cpu', dtype=dtype)
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        train_loader = FastTensorDataLoader(train_x, train_y, batch_size=min(args.D, args.batch_size), shuffle=True)
        train_iter = cycle(train_loader)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    log_steps = list()
    accuracies = list()
    pred_losses = list()
    sparsity_losses = list()
    losses = list()
    pred_losses_subtasks = dict()
    sparsity_losses_subtasks = dict()
    accuracies_subtasks = dict()
    for i in range(args.n_tasks):
        pred_losses_subtasks[i] = list()
        sparsity_losses_subtasks[i] = list()
        accuracies_subtasks[i] = list()
    early_stop_triggers = []
    
    for step in tqdm(range(args.steps), disable=not args.verbose):
        if step % args.log_freq == 0:
            with torch.no_grad():
                x_i, y_i = get_batch(n_tasks=args.n_tasks, n=args.n, Ss=Ss, codes=list(range(args.n_tasks)), sizes=test_batch_sizes, device=device, dtype=dtype)
                h_i = mlp[:2](x_i)
                y_i_pred = mlp[2](h_i)
                labels_i_pred = torch.argmax(y_i_pred, dim=1)
                accuracies.append(torch.sum(labels_i_pred == y_i).item() / args.test_points)
                sparsity_loss = torch.sum(h_i, dim=1).mean()
                sparsity_losses.append(sparsity_loss.item())
                pred_loss = loss_fn(y_i_pred, y_i)
                pred_losses.append(pred_loss.item())
                loss = pred_loss + args.l1 * sparsity_loss
                losses.append(loss)
                for i in range(args.n_tasks):
                    x_i, y_i = get_batch(n_tasks=args.n_tasks, n=args.n, Ss=[Ss[i]], codes=[i], sizes=[args.test_points_per_task], device=device, dtype=dtype)
                    h_i = mlp[:2](x_i)
                    y_i_pred = mlp[2](h_i)
                    pred_loss_i = loss_fn(y_i_pred, y_i)
                    pred_losses_subtasks[i].append(pred_loss_i.item())
                    sparsity_loss_i = torch.sum(h_i, dim=1).mean()
                    sparsity_losses_subtasks[i].append(sparsity_loss_i.item())
                    labels_i_pred = torch.argmax(y_i_pred, dim=1)
                    accuracies_subtasks[i].append(torch.sum(labels_i_pred == y_i).item() / args.test_points_per_task)
                log_steps.append(step)
                wandb_metrics = {
                    'accuracy': accuracies[-1],
                    'loss': losses[-1],
                    'pred_loss': pred_losses[-1],
                    'sparsity_loss': sparsity_losses[-1],
                }
                for i in range(args.n_tasks):
                    wandb_metrics[f'subtask_{i}_accuracy'] = accuracies_subtasks[i][-1]
                    wandb_metrics[f'subtask_{i}_pred_loss'] = pred_losses_subtasks[i][-1]
                    wandb_metrics[f'subtask_{i}_sparsity_loss'] = sparsity_losses_subtasks[i][-1]
                if use_wandb:
                    wandb.log(wandb_metrics, step=step)
            if args.stop_early:
                if step > 4000 and len(losses) >= 2 and losses[-1] > losses[-2]:
                    early_stop_triggers.append(True)
                else:
                    early_stop_triggers.append(False)
                if len(early_stop_triggers) > 10 and all(early_stop_triggers[-10:]):
                    break
                early_stop_triggers = early_stop_triggers[-10:]
                
        optimizer.zero_grad()
        if args.D == -1:
            samples = np.searchsorted(cdf, np.random.rand(args.batch_size,))
            hist, _ = np.histogram(samples, bins=args.n_tasks, range=(0, args.n_tasks-1))
            x, y_target = get_batch(n_tasks=args.n_tasks, n=args.n, Ss=Ss, codes=list(range(args.n_tasks)), sizes=hist, device=device, dtype=dtype)
        else:
            x, y_target = next(train_iter)
        h = mlp[:2](x) 
        y_pred = mlp[2](h)
        pred_loss = loss_fn(y_pred, y_target)
        sparsity_loss = torch.sum(h, dim=1).mean()
        loss = pred_loss + args.l1 * sparsity_loss
        loss.backward()
        optimizer.step()

    if args.save_dir is None:
        save_dir = "." if not use_wandb else os.path.join("/om/user/ericjm/results", wandb.run.id)
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    torch.save(mlp.state_dict(), os.path.join(save_dir, "model.pt"))
    # if use_wandb:
    #     wandb.save(str(model_path))
        
    results = {
        'log_steps': log_steps, 
        'accuracies': accuracies,
        'pred_losses': pred_losses,
        'sparsity_losses': sparsity_losses,
        'losses': losses,
        'accuracies_subtasks': accuracies_subtasks,
        'pred_losses_subtasks': pred_losses_subtasks,
        'sparsity_losses_subtasks': sparsity_losses_subtasks,
        'Ss': Ss,
    }

    with open(os.path.join(save_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f)
   
    return results
        
if __name__ == '__main__':
    main()

