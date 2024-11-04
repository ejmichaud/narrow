"""
Trains SAEs on the activations of an MLP trained on the multitask sparse
parity task. Assumes MLPs were trained with `sparse-parity-v6.py` and that
networks have one hidden layer. Uses the TopK SAE architecture.
"""


import os
import argparse
from collections import defaultdict
from itertools import islice, product
import random
import json
import pickle
import time

import torch
from colorama import Fore, Style, init
import random

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn

from sparse_autoencoder import Autoencoder
from sparse_autoencoder.model import ACTIVATIONS_CLASSES
from sparse_autoencoder.loss import normalized_mean_squared_error
# from sparse_autoencoder.train import unit_norm_decoder_, unit_norm_decoder_grad_adjustment_

import wandb


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

def rgb(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"

def visualize_msp(x, n_tasks, n, Ss, max_print=20):
    """
    Visualize a batch of multitask sparse parity vectors using an extended color palette.

    Parameters:
    -----------
    x : torch.Tensor
        Batch of input vectors of shape (B, n_tasks + n)
    n_tasks : int
        Number of tasks
    n : int
        Bit string length for sparse parity problem
    Ss : list of lists of ints
        Subsets of [1, ..., n] to compute sparse parities on
    max_print : int, optional
        Maximum number of vectors to print (default: 20)

    Returns:
    --------
    None (prints the visualization)
    """
    init(autoreset=True)  # Initialize colorama

    # Extended color palette
    colors = [
        Fore.RED, Fore.GREEN, Fore.BLUE, Fore.YELLOW, Fore.MAGENTA, Fore.CYAN,
        Fore.LIGHTRED_EX, Fore.LIGHTGREEN_EX, Fore.LIGHTBLUE_EX, Fore.LIGHTYELLOW_EX, Fore.LIGHTMAGENTA_EX, Fore.LIGHTCYAN_EX,
        rgb(255, 128, 0),  # Orange
        rgb(128, 0, 128),  # Purple
        rgb(0, 128, 128),  # Teal
        rgb(128, 128, 0),  # Olive
        rgb(255, 0, 255),  # Fuchsia
        rgb(0, 255, 255),  # Aqua
        rgb(128, 0, 0),    # Maroon
        rgb(0, 128, 0),    # Dark Green
    ]
    task_colors = [colors[i % len(colors)] for i in range(n_tasks)]

    batch_size = x.shape[0]
    to_print = min(batch_size, max_print)

    for i in range(to_print):
        if i == to_print // 2 and batch_size > max_print:
            print("...")
            i = batch_size - (to_print - to_print // 2)

        vector = x[i]
        task_index = torch.argmax(vector[:n_tasks]).item()
        task_color = task_colors[task_index]
        relevant_bits = Ss[task_index]

        # Print task indicator and bit string
        for j in range(n_tasks + n):
            if j < n_tasks:
                if j == task_index:
                    print(f"{task_color}{vector[j].int().item()}{Style.RESET_ALL}", end="")
                else:
                    print(f"{vector[j].int().item()}", end="")
            else:
                if j - n_tasks in relevant_bits:
                    print(f"{task_color}{vector[j].int().item()}{Style.RESET_ALL}", end="")
                else:
                    print(f"{vector[j].int().item()}", end="")
        print()  # New line after each vector

    if batch_size > max_print:
        print(f"... (total {batch_size} vectors in batch)")


# n_tasks = 10
# n = 20
# Ss = [
#     (i, i+1, i+2) for i in range(10)
# ]

# x, y = get_batch(n_tasks, n, Ss, codes=list(range(n)), sizes=[1]*n_tasks)
# visualize_msp(x, n_tasks, n, Ss)

def unit_norm_decoder_(autoencoder: Autoencoder) -> None:
    """
    Unit normalize the decoder weights of an autoencoder.
    """
    autoencoder.decoder.weight.data /= autoencoder.decoder.weight.data.norm(dim=0)


# def unit_norm_decoder_grad_adjustment_(autoencoder) -> None:
#     """project out gradient information parallel to the dictionary vectors - assumes that the decoder is already unit normed"""

#     assert autoencoder.decoder.weight.grad is not None

#     triton_add_mul_(
#         autoencoder.decoder.weight.grad,
#         torch.einsum("bn,bn->n", autoencoder.decoder.weight.data, autoencoder.decoder.weight.grad),
#         autoencoder.decoder.weight.data,
#         c=-1,
#     )


def main():

    p = argparse.ArgumentParser(description='Trains an SAE on the hidden activations of an MLP trained with `sparse-parity-v6.py`.')
    p.add_argument("--model-dir", type=str, required=True,
                   help="directory containing the trained model, its config, etc.")
    p.add_argument("--save_dir", type=str, required=True,
                   help="directory to save the SAE")
    p.add_argument("--n_latents", type=int, default=4096,
                   help="SAE hidden width")
    p.add_argument("--activation", type=str, default='ReLU',
                   choices=['ReLU', 'TopK'],
                   help="activation function")
    p.add_argument("--k", type=int, default=16, required=False, 
                   help="number of top activations to keep")
    p.add_argument("--steps", type=int, default=100_000,
                   help="number of training steps")
    p.add_argument("--batch-size", type=int, default=8192,
                     help="training batch size")
    p.add_argument("--normalize", action='store_true',
                   help="whether to normalize the input to the SAE")
    p.add_argument('--lr', type=float, default=1e-3,
                   help='learning rate')
    p.add_argument('--device', type=str, default=None,
                   help='device to use')
    p.add_argument('--dtype', type=str, default='float32',
                   choices=['bfloat16', 'float16', 'float32', 'float64'],  
                   help='data type')
    p.add_argument('--log-freq', type=int, default=None,
                   help='log frequency (default log_freq=steps//1000)')
    p.add_argument('--seed', type=int, default=0,
                   help='random seed')
    p.add_argument('--wandb-project', type=str, default=None,
                   help='wandb project name (omit to not log)')
    p.add_argument('--verbose', action='store_true',
                   help='whether to print progress updates')

    args = p.parse_args()


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
    
    with open(os.path.join(args.model_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    config = argparse.Namespace(**config)

    with open(os.path.join(args.model_dir, 'results.pkl'), 'rb') as f:
        Ss = pickle.load(f)['Ss']

    assert config.depth == 2, 'Only supports MLPs with depth 2'

    # create MLP
    if config.activation == 'ReLU':
        activation_fn = nn.ReLU
    elif config.activation == 'Tanh':
        activation_fn = nn.Tanh
    elif config.activation == 'Sigmoid':
        activation_fn = nn.Sigmoid
    else:
        raise ValueError(f'Invalid activation function: {config.activation}')
    
    # create model
    layers = []
    for i in range(config.depth):
        if i == 0:
            layers.append(nn.Linear(config.n_tasks + config.n, config.width))
            layers.append(activation_fn())
        elif i == config.depth - 1:
            layers.append(nn.Linear(config.width, 2))
        else:
            layers.append(nn.Linear(config.width, config.width))  
            layers.append(activation_fn())
    mlp = nn.Sequential(*layers).to(dtype).to(device)

    state_dict = torch.load(os.path.join(args.model_dir, 'model.pt'))

    mlp.load_state_dict(state_dict)
    mlp = mlp.to(dtype).to(device)

    if args.activation == 'TopK':
        activation = ACTIVATIONS_CLASSES[args.activation](args.k)
    else:
        activation = ACTIVATIONS_CLASSES[args.activation]()
    
    sae = Autoencoder(args.n_latents, config.width, 
                activation=activation,
                normalize=args.normalize).to(dtype).to(device)

    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)

    probs = np.array([n ** (-config.alpha) for n in range(1+config.offset, config.n_tasks+config.offset+1)])
    probs = probs / np.sum(probs)
    cdf = np.cumsum(probs)

    if args.wandb_project is not None:
        wandb.init(
            project=args.wandb_project, 
            config=vars(args),
            save_code=True,
        )
    
    losses = []
    steps = []

    for step in tqdm(range(args.steps), disable=not args.verbose):

        # sample batch
        samples = np.searchsorted(cdf, np.random.rand(args.batch_size,))
        hist, _ = np.histogram(samples, bins=config.n_tasks, range=(0, config.n_tasks-1))
        x, _ = get_batch(n_tasks=config.n_tasks, n=config.n, Ss=Ss, codes=list(range(config.n_tasks)), sizes=hist, device=device, dtype=dtype)

        with torch.no_grad():
            h = mlp[:2](x) # post-activation

        optimizer.zero_grad()
        _, _, h_hat = sae(h)
        loss = normalized_mean_squared_error(h_hat, h)
        loss.backward()

        unit_norm_decoder_(sae)
        # unit_norm_decoder_grad_adjustment_(sae)

        optimizer.step()

        if step % args.log_freq == 0:
            steps.append(step)
            losses.append(loss.item())
            if args.wandb_project is not None:
                wandb.log({'loss': loss.item()}, step=step)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    torch.save(sae.state_dict(), os.path.join(save_dir, "model.pt"))
    # if use_wandb:
    #     wandb.save(str(model_path))
        
    results = {
        'steps': steps, 
        'losses': losses,
    }

    with open(os.path.join(save_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f)

    return results 


if __name__ == "__main__":
    main()
