import os
import sys
import copy
import pickle
from collections import defaultdict, OrderedDict

import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from nnsight import NNsight

# add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from general.train import (
    depth, width, activation_fn,
    n_tasks, n, Ss, codes,
    get_batch, construct_mlp
)


def prunability(mlp, x, y, loss_fn, effect_thresholds=[1e-10, 1e-8, 1e-6, 1e-4, 1e-2], loss_threshold=1e-2):
    """ Computes the number of neurons that can be pruned from the MLP while 
    keeping the loss below a certain threshold on the given dataset.

    This function first estimates the effects of ablating each neuron with
    attribution patching. Then we prune neurons with effect below varying 
    thresholds and recompute the loss, checking that it remains below the
    `loss_threshold`.
    """
    mlp_n = NNsight(mlp)
    effect_thresholds = sorted(effect_thresholds)
    layer_names = [name for name, module in mlp.named_children() if name.startswith("act")]

    # compute effects
    with mlp_n.trace(x):
        activations = {}
        grads = {}
        for name in layer_names:
            activations[name] = getattr(mlp_n, name).output.save()
            grads[name] = getattr(mlp_n, name).output.grad.save()
        y_pred = mlp_n.output
        loss = loss_fn(y_pred, y)
        loss.backward()
    effects = {}
    for name in layer_names:
        effects[name] = torch.mean(activations[name] * grads[name], dim=0)
    
    # prune neurons and check loss
    pruning_results = {}
    for threshold in effect_thresholds:
        with mlp_n.trace(x):
            for name in layer_names:
                getattr(mlp_n, name).output[:, effects[name].abs() < threshold] = 0.0
            y_pred = mlp_n.output.save()
        loss = loss_fn(y_pred, y).item()
        if loss > loss_threshold:
            break
        pruning_results[threshold] = {
            "loss": loss,
            "neurons_kept": {
                name: (effects[name].abs() >= threshold).nonzero().flatten().tolist()
                for name in layer_names
            }
        }
    
    # free memory
    del mlp_n
    del activations
    del grads
    del y_pred
    del effects

    if not pruning_results:
        return {
           "loss": loss_fn(mlp.forward(x), y).item(),
              "neurons_kept": {
                name: list(range(getattr(mlp, name.replace("act", "fc")).out_features))
                for name in layer_names
              } 
        }
    # last acceptable threshold
    last_threshold = max(pruning_results.keys())
    return pruning_results[last_threshold]

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dtype = torch.float32
    torch.set_default_dtype(dtype)

    mlp = construct_mlp(
        [n_tasks + n] + [width] * (depth - 1) + [2], 
        activation_fn
    ).to(dtype).to(device)
    mlp.load_state_dict(torch.load("../general/model_final.pt"))

    LAMBDA = 3e-4

    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3, eps=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    steps = []
    losses = []
    pred_losses = []
    sparsity_losses = []
    remaining_neurons = []
    for step in tqdm(range(10_000)):
        x, y = get_batch(n_tasks, n, Ss, [[0, 1]], [2_000], device=device, dtype=dtype)
        y_pred = mlp(x)
        pred_loss = loss_fn(y_pred, y)
        sparsity_loss = 0
        for p in mlp.parameters():
            sparsity_loss += torch.sum(p.abs())
        loss = pred_loss + LAMBDA * sparsity_loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        steps.append(step)
        losses.append(loss.item())
        pred_losses.append(pred_loss.item())
        sparsity_losses.append(sparsity_loss.item())

        if step % 30 == 0:
            prunability_results = prunability(
                mlp,
                x, y,
                loss_fn,
                effect_thresholds=[1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
                loss_threshold=1e-2
            )
            n_remaining = sum(len(neurons) for neurons in prunability_results["neurons_kept"].values())
            remaining_neurons.append(n_remaining)

    # save results
    with open("results.pkl", "wb") as f:
        pickle.dump({
            "steps": steps,
            "losses": losses,
            "pred_losses": pred_losses,
            "sparsity_losses": sparsity_losses,
            "remaining_neurons": remaining_neurons,
            "remaining_neuerons_steps": list(range(0, 10_000, 30))
        }, f)
    torch.save(mlp.state_dict(), "model_final.pt")
