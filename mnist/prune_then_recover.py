#!/usr/bin/env python3
"""
Implements attribute-based pruning for MNIST model followed by recovery training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from collections import defaultdict
from tqdm.auto import tqdm
import numpy as np
import json
from torch.optim.lr_scheduler import CosineAnnealingLR
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model definition (identical to pruning.py)
class Net(nn.Module):
    def __init__(
        self,
        input_dim=28 * 28,
        hidden_dims=[1200, 1200],
        output_dim=10,
    ):
        super(Net, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.masks = []  # Store binary masks for each layer
        previous_dim = input_dim

        for h_dim in hidden_dims:
            self.fc_layers.append(nn.Linear(previous_dim, h_dim))
            # Initialize mask with ones - makes one mask for each hidden layer
            self.masks.append(torch.ones(h_dim, device=device))
            previous_dim = h_dim

        self.output_layer = nn.Linear(previous_dim, output_dim)
        self.masks.append(torch.ones(output_dim, device=device))

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        for i, fc in enumerate(self.fc_layers):
            # Apply mask to weights before forward pass
            masked_weight = fc.weight * self.masks[i].unsqueeze(1)
            x = F.linear(x, masked_weight, fc.bias)
            x = F.relu(x)
        x = self.output_layer(x)
        return x

    def update_masks(self, neuron_indices_to_prune):
        """
        Update masks based on provided indices to prune

        Args:
            neuron_indices_to_prune: dict mapping layer index to list of neuron indices to prune
        """
        active_neurons = []
        for i in range(len(self.fc_layers)):
            if i in neuron_indices_to_prune:
                for neuron_idx in neuron_indices_to_prune[i]:
                    self.masks[i][neuron_idx] = 0.0
                    # Zero out the weights too
                    self.fc_layers[i].weight.data[neuron_idx] = 0.0
                    if i < len(self.fc_layers) - 1:
                        # Zero out the weights in the next layer that connect to this neuron
                        self.fc_layers[i + 1].weight.data[:, neuron_idx] = 0.0
                    else:
                        # Zero out the weights in the output layer that connect to this neuron
                        self.output_layer.weight.data[:, neuron_idx] = 0.0

            active_neurons.append(int(self.masks[i].sum().item()))

        return active_neurons


def get_attribution_hook(cache, name, hook_cache):
    """
    A hook to compute attributions. Registers at each layer.
    """

    def attribution_hook(module, input, output):
        def backward_hook(grad):
            # Compute attribution: element-wise product of activations and gradients
            activation = output.detach()
            attribution = activation * grad
            cache[name] = attribution
            return grad

        hook = output.register_hook(backward_hook)
        hook_cache[name] = hook
        return None

    return attribution_hook


def compute_attribution_scores(model, train_loader, num_batches=10):
    """
    Compute attribution scores for neurons in the model

    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        num_batches: Number of batches to compute attribution on

    Returns:
        List of (layer_idx, neuron_idx, score) tuples, sorted by score (lowest first)
    """
    model.eval()
    scores = {
        i: torch.zeros(model.masks[i].size(0), device=device)
        for i in range(len(model.fc_layers))
    }
    total_samples = 0

    for batch_idx, (data, target) in enumerate(
        tqdm(train_loader, desc="Computing attribution scores")
    ):
        if batch_idx >= num_batches:
            break

        data, target = data.to(device), target.to(device)

        # Set up hooks for attribution
        cache = {}
        forward_hooks = {}
        backward_handles = {}

        # Register hooks for each layer's ReLU activation
        for i, layer in enumerate(model.fc_layers):
            forward_hooks[i] = layer.register_forward_hook(
                get_attribution_hook(cache, i, backward_handles)
            )

        # Forward and backward pass
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()

        # Aggregate attribution scores (use absolute values)
        for i in range(len(model.fc_layers)):
            if i in cache:
                # Sum attribution across batch and spatial dimensions
                layer_attr = cache[i]
                # Take mean absolute value across batch and input dimensions
                abs_attr = layer_attr.abs().mean(dim=tuple(range(layer_attr.ndim - 1)))
                scores[i] += abs_attr

            # Remove hooks
            forward_hooks[i].remove()
            if i in backward_handles:
                backward_handles[i].remove()

        total_samples += 1

    # Normalize scores by number of samples
    for i in scores:
        scores[i] /= total_samples

    # Create list of (layer_idx, neuron_idx, score) tuples
    score_tuples = []
    for layer_idx in range(len(model.fc_layers)):
        for neuron_idx in range(scores[layer_idx].size(0)):
            # Only consider neurons that are not already pruned
            if model.masks[layer_idx][neuron_idx] > 0:
                score_tuples.append(
                    (layer_idx, neuron_idx, scores[layer_idx][neuron_idx].item())
                )

    # Sort by score (ascending order - lowest scores will be pruned first)
    score_tuples.sort(key=lambda x: x[2])

    return score_tuples


def apply_pruning(model, score_tuples, target_neurons):
    """
    Apply pruning based on attribution scores to reach target number of neurons,
    distributing pruning evenly across layers.

    Args:
        model: The neural network model
        score_tuples: List of (layer_idx, neuron_idx, score) tuples
        target_neurons: Target number of active neurons to keep

    Returns:
        Dictionary containing pruning statistics
    """
    # Calculate current active neurons
    layer_active_counts = [int(mask.sum().item()) for mask in model.masks]
    total_active = sum(layer_active_counts)

    # Calculate how many neurons to prune
    neurons_to_prune = total_active - target_neurons
    if neurons_to_prune <= 0:
        print(f"Already at or below target: {total_active} <= {target_neurons}")
        return {
            "total_active_before": total_active,
            "total_active_after": total_active,
            "neurons_pruned": 0,
        }

    print(f"Pruning {neurons_to_prune} neurons to reach target of {target_neurons}")

    # Group score tuples by layer
    layer_scores = {}
    for layer_idx, neuron_idx, score in score_tuples:
        if layer_idx not in layer_scores:
            layer_scores[layer_idx] = []
        layer_scores[layer_idx].append((neuron_idx, score))

    # Sort each layer's neurons by score (ascending)
    for layer_idx in layer_scores:
        layer_scores[layer_idx].sort(key=lambda x: x[1])

    # Calculate pruning ratios per layer based on current layer sizes
    # We want to prune approximately the same percentage from each layer
    pruning_ratio = neurons_to_prune / total_active
    neurons_to_prune_per_layer = {
        i: int(layer_active_counts[i] * pruning_ratio)
        for i in range(len(layer_active_counts))
    }

    # Adjust to ensure we prune exactly the desired number
    total_to_prune = sum(neurons_to_prune_per_layer.values())
    if total_to_prune < neurons_to_prune:
        # Distribute remaining neurons to prune across layers
        remaining = neurons_to_prune - total_to_prune
        layer_indices = sorted(
            range(len(layer_active_counts)),
            key=lambda i: layer_active_counts[i],
            reverse=True,
        )
        for i in range(remaining):
            # Add one more neuron to prune to each layer, starting with the largest
            layer_idx = layer_indices[i % len(layer_indices)]
            neurons_to_prune_per_layer[layer_idx] += 1

    # Create a dictionary to collect neuron indices to prune per layer
    indices_to_prune = defaultdict(list)

    # Add neurons to prune from each layer
    for layer_idx, layer_tuples in layer_scores.items():
        # Limit to the pre-calculated number for this layer
        to_prune_count = min(
            neurons_to_prune_per_layer.get(layer_idx, 0), len(layer_tuples)
        )

        # Safety check: ensure we leave at least a small number of neurons in each layer
        # but still allow reaching the target total neuron count
        # Ensure each layer keeps at least ~target_neurons/num_layers neurons but not more than necessary
        num_layers = len(layer_active_counts)
        target_per_layer = max(1, target_neurons // num_layers)
        min_to_keep = min(
            target_per_layer, max(1, int(layer_active_counts[layer_idx] * 0.1))
        )
        max_to_prune = layer_active_counts[layer_idx] - min_to_keep
        to_prune_count = min(to_prune_count, max_to_prune)

        # Add the lowest scoring neurons to the pruning list
        for i in range(to_prune_count):
            neuron_idx, _ = layer_tuples[i]
            indices_to_prune[layer_idx].append(neuron_idx)

    # Apply pruning
    active_neurons = model.update_masks(indices_to_prune)
    total_active_after = sum(active_neurons)
    total_pruned = sum(len(indices) for indices in indices_to_prune.values())

    # Print statistics
    print(f"Active neurons per layer after pruning: {active_neurons}")
    print(f"Total active neurons: {total_active_after}")

    return {
        "total_active_before": total_active,
        "total_active_after": total_active_after,
        "neurons_pruned": total_pruned,
        "neurons_pruned_per_layer": {k: len(v) for k, v in indices_to_prune.items()},
    }


# Training function
def train(
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    max_grad_norm=1.0,
    total_datapoints=0,
    partial_epoch=False,
    segment_index=0,  # 0-15 for each 1/16 of epoch
):
    """
    Train the model for one epoch or a segment of an epoch
    """
    # Simplified segment naming
    segment_name = f"segment {segment_index+1}/16"

    print(
        f"\nStarting {segment_name} of training at epoch={epoch}, total_datapoints={total_datapoints}"
    )

    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    # Calculate how many batches to process for this segment of epoch
    total_batches = len(train_loader)
    segment_batches = total_batches // 16

    # Determine which batches to process
    start_batch = segment_index * segment_batches
    end_batch = (
        (segment_index + 1) * segment_batches if segment_index < 15 else total_batches
    )

    for batch_idx, (data, target) in enumerate(train_loader, 0):
        # Skip batches not in our segment
        if partial_epoch and (batch_idx < start_batch or batch_idx >= end_batch):
            continue

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # Compute loss (no pruning penalty)
        loss = F.cross_entropy(output, target)

        # Backward pass and optimization
        loss.backward()

        # Zero out gradients for pruned weights
        for i, layer in enumerate(model.fc_layers):
            if layer.weight.grad is not None:
                layer.weight.grad.data *= model.masks[i].unsqueeze(1)

        # Gradient clipping (match pruning.py)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += data.size(0)

    current_datapoints = total_datapoints + total_samples
    accuracy = 100.0 * correct / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0

    # Calculate active neurons
    active_neurons = []
    for i, layer in enumerate(model.fc_layers):
        active_count = int(model.masks[i].sum().item())
        active_neurons.append(active_count)
    total_active = sum(active_neurons)

    segment_str = f"segment {segment_index+1}/16"
    print(
        f"Train Epoch: {epoch} {segment_str} [{total_samples}/{len(train_loader.dataset)//16 if partial_epoch else len(train_loader.dataset)} "
        f"({100. * total_samples / (len(train_loader.dataset)//16 if partial_epoch else len(train_loader.dataset)):.0f}%)]\t"
        f"Loss: {avg_loss:.6f}\tAccuracy: {accuracy:.2f}%"
    )
    print(f"Active neurons per layer: {active_neurons}")
    print(f"Total active neurons: {total_active}")
    print(f"Total datapoints processed: {current_datapoints}")

    return current_datapoints, accuracy, active_neurons


def test(model, device, test_loader):
    """
    Test the model - identical to pruning.py
    """
    model.eval()
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)

    accuracy = 100.0 * correct / total_samples
    print(f"Test set: Accuracy: {correct}/{total_samples} ({accuracy:.2f}%)")
    return accuracy


def filter_even_digits(dataset):
    """Filter the dataset to include only even digits."""
    even_indices = [
        i for i, (_, target) in enumerate(dataset) if target in {0, 2, 4, 6, 8}
    ]
    return torch.utils.data.Subset(dataset, even_indices)


def save_accuracy_data(model_accuracies, filename="data/attribution_pruning.csv"):
    """Saves the accuracy data to a CSV file."""
    os.makedirs("data", exist_ok=True)
    csv_path = filename

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as csvfile:
        fieldnames = [
            "Active Neurons",
            "Learning Rate",
            "Accuracy",
            "Epoch",
            "Datapoints",
            "Target Neurons",
            "Success",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for entry in model_accuracies:
            writer.writerow(entry)

    entry_count = len(model_accuracies)
    if entry_count == 1:
        print(f"Saved 1 result to {csv_path}")
    else:
        print(f"Saved {entry_count} results to {csv_path}")


def main():
    batch_size = 16
    hidden_dim = 1200
    target_datapoints = 50000
    target_accuracy = 97.0  # Target accuracy threshold
    max_grad_norm = 1.0
    even_digits_only = True
    check_partial_epochs = True  # Enable 1/16-epoch checking

    # Create a list of target neuron counts to evaluate
    target_neurons_list = [600, 190, 25]

    learning_rates = [4e-4, 1e-3, 3e-3, 6e-3]

    # Number of runs per configuration
    num_runs = 10

    # Maximum datapoints before giving up
    max_datapoints = 50000

    # Data loading & preprocessing
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Filter datasets to include only even digits if specified
    if even_digits_only:
        train_dataset = filter_even_digits(train_dataset)
        test_dataset = filter_even_digits(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create output directory
    os.makedirs("results_attribution", exist_ok=True)

    # Results to store
    all_results = []
    best_configs_by_target = {}

    # Initialize total_datapoints to include attribution computation cost
    attribution_datapoints = 30 * batch_size  # 30 batches * 16 = 480 datapoints
    total_datapoints = attribution_datapoints

    # Run for each target neuron count
    for target_neurons in target_neurons_list:
        print(f"\n{'='*80}")
        print(f"STARTING ATTRIBUTION PRUNING FOR TARGET NEURONS: {target_neurons}")
        print(f"{'='*80}")

        sweep_results = []

        # Track best hyperparameters and results for this target
        best_config = {
            "lr": None,
            "accuracy": 0.0,
            "active_neurons": float("inf"),
            "epochs": 0,
            "datapoints": float("inf"),
            "success": False,
            "target_neurons": target_neurons,
        }

        for lr in learning_rates:
            print(f"\n{'-'*80}")
            print(
                f"Pruning with Target neurons = {target_neurons}, Learning Rate = {lr}"
            )
            print(f"{'-'*80}")

            # Lists to store results across multiple runs
            run_active_neurons = []
            run_accuracies = []
            run_epochs = []
            run_datapoints = []
            run_success = []

            # Run each configuration multiple times
            for run in range(num_runs):
                print(f"\nRun {run+1}/{num_runs}")
                print(f"{'-'*40}")

                # Load the original pretrained model
                original_model_path = "models/original_model.pth"
                if not os.path.exists(original_model_path):
                    raise FileNotFoundError(
                        "Please train the original model first using train_original.py"
                    )

                # Initialize model with the same architecture
                model = Net(
                    hidden_dims=[hidden_dim, hidden_dim],
                ).to(device)
                model.load_state_dict(torch.load(original_model_path))

                # Test original model
                original_accuracy = test(model, device, test_loader)
                print(f"Original model accuracy: {original_accuracy:.2f}%")

                # Compute attribution scores
                score_tuples = compute_attribution_scores(
                    model, train_loader, num_batches=30
                )

                # Apply pruning to reach target neuron count
                pruning_stats = apply_pruning(model, score_tuples, target_neurons)

                # Test the pruned model before training
                pruned_accuracy = test(model, device, test_loader)
                print(f"Pruned model accuracy before training: {pruned_accuracy:.2f}%")

                # Now train the pruned model until reaching target accuracy
                optimizer = optim.Adam(model.parameters(), lr=lr)

                # Use CosineAnnealingLR like in pruning.py
                dataset_size = len(train_loader.dataset)
                t_max = max(1, target_datapoints // dataset_size)
                scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=lr)

                total_datapoints = 0
                epoch = 0

                # Variables to track our targets
                accuracy_target_reached = False
                current_accuracy = 0.0

                # Train until we've reached target accuracy or hit max datapoints
                while not accuracy_target_reached and total_datapoints < max_datapoints:
                    epoch += 1
                    scheduler.step(epoch)

                    # Go through each 1/16 of the epoch
                    for segment in range(16):
                        if not check_partial_epochs and segment > 0:
                            continue  # Skip segments 1-15 if not checking partial epochs

                        # Train on this segment of the data
                        current_datapoints, train_accuracy, active_neurons = train(
                            model,
                            device,
                            train_loader,
                            optimizer,
                            epoch,
                            max_grad_norm=max_grad_norm,
                            total_datapoints=total_datapoints,
                            partial_epoch=check_partial_epochs,
                            segment_index=segment,
                        )

                        total_datapoints = current_datapoints
                        total_active_neurons = sum(active_neurons)

                        # Test after this segment
                        current_accuracy = test(model, device, test_loader)

                        # Check if target accuracy reached
                        accuracy_target_reached = current_accuracy >= target_accuracy

                        if accuracy_target_reached:
                            print(
                                f"SUCCESS! Reached target accuracy at segment {segment+1} of epoch {epoch}: {current_accuracy:.2f}%"
                            )
                            break  # Exit the segment loop

                    # If we reached target accuracy, exit the epoch loop too
                    if accuracy_target_reached:
                        break

                # Final status message for this run
                success = accuracy_target_reached
                if success:
                    print(
                        f"Run {run+1}: Successfully reached target accuracy after {epoch} epochs and {total_datapoints} datapoints"
                    )
                else:
                    print(
                        f"Run {run+1}: Failed to reach target accuracy after {epoch} epochs and {total_datapoints} datapoints"
                    )
                    print(
                        f"Final state: {pruning_stats['total_active_after']} neurons with {current_accuracy:.2f}% accuracy"
                    )

                # Store results for this run
                run_active_neurons.append(pruning_stats["total_active_after"])
                run_accuracies.append(current_accuracy)
                run_epochs.append(epoch)
                run_datapoints.append(total_datapoints)
                run_success.append(success)

            # Calculate average statistics across all runs
            avg_active_neurons = sum(run_active_neurons) / num_runs
            avg_accuracy = sum(run_accuracies) / num_runs
            avg_epochs = sum(run_epochs) / num_runs
            avg_datapoints = sum(run_datapoints) / num_runs
            success_rate = sum(run_success) / num_runs

            # Print summary of all runs
            print(f"\n{'-'*80}")
            print(f"SUMMARY OF {num_runs} RUNS FOR LR={lr}")
            print(f"{'-'*80}")
            print(
                f"{'Run':<5} {'Active':<10} {'Accuracy':<10} {'Epochs':<10} {'Datapoints':<12} {'Success':<8}"
            )
            print(f"{'-'*80}")

            for i in range(num_runs):
                print(
                    f"{i+1:<5} {run_active_neurons[i]:<10} {run_accuracies[i]:<10.2f} "
                    f"{run_epochs[i]:<10} {run_datapoints[i]:<12} "
                    f"{'✓' if run_success[i] else '✗'}"
                )

            print(f"{'-'*80}")
            print(
                f"AVG:  {avg_active_neurons:<10.1f} {avg_accuracy:<10.2f} "
                f"{avg_epochs:<10.1f} {avg_datapoints:<12.1f} "
                f"{success_rate*100:<7.1f}%"
            )
            print(f"{'-'*80}")

            # Save average results for this configuration
            result = {
                "Active Neurons": int(avg_active_neurons),
                "Learning Rate": lr,
                "Accuracy": avg_accuracy,
                "Epoch": avg_epochs,
                "Datapoints": avg_datapoints,
                "Target Neurons": target_neurons,
                "Success": success_rate
                >= 0.5,  # Success if at least half the runs succeeded
            }

            # Add to sweep results
            sweep_results.append(result)

            # Add to results collection
            all_results.append(result)

            # Check if this is the best configuration for current target
            is_better = False
            if result["Success"] and not best_config["success"]:
                is_better = True
            elif result["Success"] == best_config["success"] and result["Success"]:
                # If both are successful, prioritize the one with fewer datapoints
                if avg_datapoints < best_config["datapoints"]:
                    is_better = True
                # If datapoints are the same, use accuracy as tiebreaker
                elif (
                    avg_datapoints == best_config["datapoints"]
                    and avg_accuracy > best_config["accuracy"]
                ):
                    is_better = True
                # If accuracy is also the same, prefer fewer active neurons
                elif (
                    avg_datapoints == best_config["datapoints"]
                    and avg_accuracy == best_config["accuracy"]
                    and avg_active_neurons < best_config["active_neurons"]
                ):
                    is_better = True

            if (
                is_better or best_config["lr"] is None
            ):  # Also update if this is the first run
                best_config["lr"] = lr
                best_config["accuracy"] = avg_accuracy
                best_config["active_neurons"] = avg_active_neurons
                best_config["epochs"] = avg_epochs
                best_config["datapoints"] = avg_datapoints
                best_config["success"] = result["Success"]

        # Store the best configuration for this target neuron count
        best_configs_by_target[target_neurons] = best_config

        # Print summary table for this target neuron count
        print(f"\n{'-'*100}")
        print(f"SUMMARY FOR TARGET NEURONS: {target_neurons}")
        print(f"{'-'*100}")
        print(
            f"{'LR':<10} {'Active':<10} {'Accuracy':<10} {'Epochs':<10} {'Datapoints':<12} {'Success':<8}"
        )
        print(f"{'-'*100}")

        for result in sweep_results:
            print(
                f"{result['Learning Rate']:<10.3e} "
                f"{result['Active Neurons']:<10} {result['Accuracy']:<10.2f} "
                f"{result['Epoch']:<10.1f} {result['Datapoints']:<12.1f} "
                f"{'✓' if result['Success'] else '✗'}"
            )

        print(f"{'-'*100}")
        print("BEST CONFIGURATION:")
        if best_config["lr"] is not None:
            print(f"LR: {best_config['lr']:.3e}")
            print(
                f"Active neurons: {best_config['active_neurons']:.1f}, Accuracy: {best_config['accuracy']:.2f}%"
            )
            print(
                f"Epochs: {best_config['epochs']:.1f}, Datapoints: {best_config['datapoints']:.1f}"
            )
            print(f"Success: {'✓' if best_config['success'] else '✗'}")
        else:
            print("No successful configuration found.")
        print(f"{'-'*100}")

        # Save the best configuration for this target neuron count to CSV
        if best_config["success"]:
            best_result = {
                "Active Neurons": int(best_config["active_neurons"]),
                "Learning Rate": best_config["lr"],
                "Accuracy": best_config["accuracy"],
                "Epoch": best_config["epochs"],
                "Datapoints": best_config["datapoints"],
                "Target Neurons": target_neurons,
                "Success": best_config["success"],
            }
            save_accuracy_data([best_result], filename="data/attribution_pruning.csv")
            print(f"\nSaved best configuration to data/attribution_pruning.csv")
        else:
            print(
                f"\nNo successful configuration found for target neurons {target_neurons}"
            )

    # Save all results
    with open("results_attribution/all_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    # Print final summary of best configurations for each target
    print(f"\n{'='*100}")
    print(f"BEST CONFIGURATIONS SUMMARY")
    print(f"{'='*100}")
    print(
        f"{'Target':<10} {'LR':<10} {'Active':<10} {'Accuracy':<10} {'Datapoints':<12} {'Success':<8}"
    )
    print(f"{'-'*100}")

    for target, config in best_configs_by_target.items():
        if config["lr"] is not None:
            print(
                f"{target:<10} {config['lr']:<10.3e} "
                f"{config['active_neurons']:<10.1f} {config['accuracy']:<10.2f} "
                f"{config['datapoints']:<12.1f} "
                f"{'✓' if config['success'] else '✗'}"
            )

        print(f"{'-'*100}")
        print("BEST CONFIGURATION:")
        if config["lr"] is not None:
            print(f"LR: {config['lr']:.3e}")
            print(
                f"Active neurons: {config['active_neurons']:.1f}, Accuracy: {config['accuracy']:.2f}%"
            )
            print(
                f"Epochs: {config['epochs']:.1f}, Datapoints: {config['datapoints']:.1f}"
            )
            print(f"Success: {'✓' if config['success'] else '✗'}")
        else:
            print("No successful configuration found.")
        print(f"{'-'*100}")


if __name__ == "__main__":
    main()
