"""Implements pruning using specified pruning loss"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import csv
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import matplotlib.pyplot as plt
import numpy as np
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRY ABLATING BY IMPORTANCE


def visualize_mlp(model, weight_threshold=0.3, color_range=[-1, 1]):
    # Check if the model is an instance of Net
    assert isinstance(model, Net), "Model must be an instance of Net"
    layers = model.fc_layers + [model.output_layer]

    fig, ax = plt.subplots(figsize=(12, 8))
    max_width = max(max(layer.in_features, layer.out_features) for layer in layers)
    ax.set_xlim(-max_width * 0.05, max_width * 1.05)
    ax.set_ylim(-0.2, len(layers) + 0.2)

    positions = list()
    for i, layer in enumerate(layers):
        pos = np.linspace(0, max_width - 1, layer.in_features)
        positions.append(pos)
    positions.append(np.linspace(0, max_width - 1, layer.out_features))

    # Plot neurons in input layer
    ax.scatter(
        positions[0],  # Use all position indices for input layer
        [0] * len(positions[0]),  # Same number of zeros as input neurons
        s=10,
        c="black",
        zorder=10,
    )
    # Plot hidden layers (only active neurons) edit
    for i, layer in enumerate(layers[1:]):
        mask = model.masks[i].cpu().numpy()
        active_positions = positions[i + 1][mask > 0]
        ax.scatter(
            active_positions,
            [i + 1] * len(active_positions),
            s=10,
            c="black",
            zorder=10,
        )

    # plot neurons in output layer
    ax.scatter(
        positions[-1],
        [len(layers)] * 10,
        s=10,
        c="black",
        zorder=10,
    )
    # plot the connections for the first layer
    layer = layers[0]
    weights = (
        layer.weight.detach().cpu().numpy()
    )  # out_features by in_features, ie 1200 by 784
    mask = model.masks[0].unsqueeze(1).cpu().numpy()  # out_features by 1
    weights = weights * mask
    for j in range(layer.in_features):  # for j in in_features
        for k in range(weights.shape[0]):  # for k in out_features
            color = plt.cm.bwr(
                (weights[k, j] - color_range[0]) / (color_range[1] - color_range[0])
            )
            ax.plot(
                [positions[0][j], positions[1][k]],
                [0, 1],
                c=color,
                linewidth=abs(weights[k, j]) * 2,
                alpha=0.7,
            )
    # plot the connections for next layer — for first, want to double mask, with masks[0].unsqueeze(1) to get mask outfeatures leaving these neurons, and also masks[1].unsqueeze(0) to mask weights into next layer
    layer = layers[1]
    weights = (
        layer.weight.detach().cpu().numpy()
    )  # out_features by in_features ie 1200 by 1200
    mask1 = model.masks[1].unsqueeze(1).cpu().numpy()  # out_features by 1
    mask2 = model.masks[0].unsqueeze(0).cpu().numpy()  # 1 by in_features
    # Apply mask
    weights = weights * mask1 * mask2
    for j in range(layer.in_features):  # for j in in_features
        for k in range(weights.shape[0]):  # for k in out_features
            color = plt.cm.bwr(
                (weights[k, j] - color_range[0]) / (color_range[1] - color_range[0])
            )
            ax.plot(
                [positions[1][j], positions[2][k]],
                [1, 2],
                c=color,
                linewidth=abs(weights[k, j]) * 2,
                alpha=0.7,
            )
    # for last layer, just want to mask the in_features, so masks[1].unsqueeze(0) gives 1 by in_features which multiplies right
    layer = layers[-1]
    weights = layer.weight.detach().cpu().numpy()
    mask = model.masks[1].unsqueeze(0).cpu().numpy()
    weights = weights * mask
    for j in range(layer.in_features):
        for k in range(weights.shape[0]):
            color = plt.cm.bwr(
                (weights[k, j] - color_range[0]) / (color_range[1] - color_range[0])
            )
            ax.plot(
                [positions[-2][j], positions[-1][k]],
                [len(layers) - 1, len(layers)],
                c=color,
                linewidth=abs(weights[k, j]) * 2,
                alpha=0.7,
            )
    ax.axis("off")


# Model definition
class Net(nn.Module):
    def __init__(
        self,
        input_dim=28 * 28,
        hidden_dims=[1200, 1200],
        output_dim=10,
        pruning_threshold=0.08,
    ):
        super(Net, self).__init__()
        self.pruning_threshold = pruning_threshold
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

    def update_masks(self):
        """Update masks based on tied l2 magnitudes"""
        active_neurons = []
        for i, layer in enumerate(self.fc_layers):
            W1 = self.fc_layers[i].weight
            if i < len(self.fc_layers) - 1:
                W2 = self.fc_layers[i + 1].weight
            else:
                W2 = self.output_layer.weight
            # find combined l2 norms
            combined = torch.cat([W1, W2.t()], dim=1)
            norms = torch.norm(combined, p=2, dim=1)
            # Update mask
            new_mask = (norms > self.pruning_threshold).float()
            self.masks[i] = self.masks[i] * new_mask
            layer.weight.data *= self.masks[i].unsqueeze(1)
            active_neurons.append(int(self.masks[i].sum().item()))

        return active_neurons


# options: l1, tied_l1, tied_l2, tied_l2_with_l2, tied_l2_with_l1, tied_l1_with_l1, tied_l2_with_lhalf, tied_l1_with_lhalf
def pruning_loss(model, penalty_type="tied_l2_with_l2"):
    penalty = 0.0
    for i in range(len(model.fc_layers)):
        W1 = model.fc_layers[i].weight
        # if we're not on the last layer
        if i != len(model.fc_layers) - 1:
            W2 = model.fc_layers[i + 1].weight
        # if we are
        elif i == len(model.fc_layers) - 1:
            W2 = model.output_layer.weight

        # L1 of entire network
        if penalty_type == "l1":
            penalty = penalty + torch.norm(W1, p=1, dim=1).sum() / W1.numel()
            if i == len(model.fc_layers) - 1:
                penalty = penalty + torch.norm(W2, p=1, dim=1).sum() / W1.numel()

        elif penalty_type == "tied_l1":
            # Concatenate W1 with transposed W2 along dim=1
            combined = torch.cat([W1, W2.t()], dim=1)
            # Take norm of each row (where each row represents all weights connected to one neuron), then divide by # of params
            penalty = penalty + (torch.norm(combined, p=1, dim=1).sum()) / (
                W1.numel() + W2.numel()
            )

        elif penalty_type == "tied_l2":
            # Concatenate W1 with transposed W2 along dim=1
            combined = torch.cat([W1, W2.t()], dim=1)
            # Take norm of each row (where each row represents all weights connected to one neuron), then divide by # of params
            penalty = penalty + (torch.norm(combined, p=2, dim=1).sum())

        # inverse L2 of L2s of tied weights
        elif penalty_type == "tied_l2_with_l2":
            combined = torch.cat([W1, W2.t()], dim=1)
            l2 = torch.norm(combined, p=2, dim=1)
            inverse_l2_of_l2 = l2 / ((torch.norm(l2, p=2, dim=0).sum()) ** 0.2)
            penalty = penalty + inverse_l2_of_l2.sum() / (W1.numel() + W2.numel())

        # inverse L1 of L2s of tied weights
        elif penalty_type == "tied_l2_with_l1":
            combined = torch.cat([W1, W2.t()], dim=1)
            l2 = torch.norm(combined, p=2, dim=1)
            inverse_l1_of_l2 = l2 / ((torch.norm(l2, p=1, dim=0).sum()) ** 0.2)
            penalty = penalty + inverse_l1_of_l2.sum() / (W1.numel() + W2.numel())

        # inverse L1 of L1s of tied weights
        elif penalty_type == "tied_l1_with_l1":
            combined = torch.cat([W1, W2.t()], dim=1)
            l1 = torch.norm(combined, p=1, dim=1)
            inverse_l1_of_l1 = l1 / ((torch.norm(l1, p=1, dim=0).sum()) ** 0.2)
            penalty = penalty + inverse_l1_of_l1.sum() / (W1.numel() + W2.numel())

        # lhalf of L2s of tied weights
        elif penalty_type == "tied_l2_with_lhalf":
            combined = torch.cat([W1, W2.t()], dim=1)
            l2 = torch.norm(combined, p=2, dim=1)
            lhalf_of_l2 = torch.norm(l2, p=0.5, dim=0)
            penalty = penalty + lhalf_of_l2 / (W1.numel() + W2.numel())

        # lhalf of L1s of tied weights
        elif penalty_type == "tied_l1_with_lhalf":
            combined = torch.cat([W1, W2.t()], dim=1)
            l1 = torch.norm(combined, p=1, dim=1)
            lhalf_of_l1 = torch.norm(l1, p=0.5, dim=0)
            penalty = penalty + lhalf_of_l1 / (W1.numel() + W2.numel())

        # # L1 of L1s of tied weights — selecting only bottom 1/3
        # elif penalty_type == "tied_l1_with_l1":
        #     combined = torch.cat([W1, W2.t()], dim=1)
        #     l1_norms = torch.norm(combined, p=1, dim=1)

        #     # Sort the L1 norms and select the bottom third
        #     sorted_indices = torch.argsort(l1_norms)
        #     num_selected = sorted_indices.size(0) // 3
        #     # Add the number of currently masked neurons
        #     num_masked = (model.masks[i] == 0).sum().item()
        #     num_selected += num_masked
        #     selected_indices = sorted_indices[:num_selected]

        #     # Calculate penalty only for the selected indices
        #     selected_l1 = l1_norms[selected_indices].sum()
        #     num_elements_in_selected = combined[selected_indices].numel()

        #     penalty = penalty + selected_l1 / num_elements_in_selected

        # elif penalty_type == "next_try":
        #     combined = torch.cat([W1, W2.t()], dim=1)
        #     l1_norms = torch.norm(combined, p=1, dim=1)

        #     # Sort the L1 norms and select the bottom third
        #     sorted_indices = torch.argsort(l1_norms)
        #     num_selected = sorted_indices.size(0) // 3
        #     # Add the number of currently masked neurons
        #     num_masked = (model.masks[i] == 0).sum().item()
        #     num_selected += num_masked
        #     selected_indices = sorted_indices[:num_selected]

        #     # Calculate L_1/2 norm for the selected indices
        #     selected_l1_half = torch.pow(l1_norms[selected_indices], 0.5).sum()
        #     num_elements_in_selected = combined[selected_indices].numel()
        #     penalty = penalty + torch.sqrt(selected_l1_half) / num_elements_in_selected

    return penalty


# Training and Testing
def train(
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    beta,
    penalty_type,
    max_grad_norm=1.0,
    total_datapoints=0,
    partial_epoch=False,
    segment_index=0,  # 0-15 for each 1/16 of epoch
):
    # Simplified segment naming
    segment_name = f"segment {segment_index+1}/16"

    print(
        f"\nStarting {segment_name} of training with beta={beta}, penalty_type={penalty_type}, epoch={epoch}, total_datapoints={total_datapoints}"
    )
    model.train()
    total_loss = 0
    total_ce_loss = 0
    correct = 0
    total_samples = 0
    total_norm = 0

    active_neurons = None  # Initialize to avoid reference before assignment
    total_active = 0  # Initialize here to avoid the UnboundLocalError

    # Calculate how many batches to process for this segment of epoch
    dataset_size = len(train_loader.dataset)
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

        # PLOTTING
        # if batch_idx % 100 == 1:
        #     visualize_mlp(
        #         model=model, weight_threshold=0.0, color_range=[-0.3, 0.3]
        #     )  # HERE
        #     plt.title(f"Step {len(os.listdir('data/weight_vis'))}")
        #     plt.savefig(
        #         f"data/weight_vis/pruning_step_{len(os.listdir('data/weight_vis')):04d}.png"
        #     )
        #     plt.close()

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # Compute loss
        loss_ce = F.cross_entropy(output, target)
        penalty = pruning_loss(model, penalty_type=penalty_type)
        detached_penalty = torch.tensor(penalty).detach()
        loss = loss_ce + beta * penalty

        # Backward pass and optimization
        loss.backward()

        # Zero out gradients of pruned weights
        for i, layer in enumerate(model.fc_layers):
            mask = model.masks[i]
            if layer.weight.grad is not None:
                layer.weight.grad.data *= mask.unsqueeze(1)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        # Update masks if we're still pruning, otherwise just calculate active neurons
        if beta > 0:
            active_neurons = model.update_masks()
        else:
            # Just calculate the active neurons without updating masks
            active_neurons = []
            for i, layer in enumerate(model.fc_layers):
                active_count = int(model.masks[i].sum().item())
                active_neurons.append(active_count)

        total_active = sum(active_neurons)

        total_loss += loss
        total_ce_loss += loss_ce
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += data.size(0)

    # Calculate final metrics for the segment
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_ce_loss = total_ce_loss / total_samples if total_samples > 0 else 0
    avg_pruning_loss = avg_loss - avg_ce_loss
    accuracy = 100.0 * correct / total_samples if total_samples > 0 else 0
    current_datapoints = total_datapoints + total_samples

    # Calculate average weight norm
    param_count = 0
    total_norm = 0
    for layer in model.fc_layers + [model.output_layer]:
        layer_norm = torch.norm(layer.weight, p=1)
        total_norm += layer_norm
        param_count += layer.weight.numel()
    avg_weight_norm = total_norm / param_count

    segment_str = f"segment {segment_index+1}/16"
    print(
        f"Train Epoch: {epoch} {segment_str} [{total_samples}/{len(train_loader.dataset)//16 if partial_epoch else len(train_loader.dataset)} "
        f"({100. * total_samples / (len(train_loader.dataset)//16 if partial_epoch else len(train_loader.dataset)):.0f}%)]\t"
        f"Loss: {avg_loss:.6f}\tAccuracy: {accuracy:.2f}%"
    )
    print(f"Active neurons per layer: {active_neurons}")
    print(f"Average weight norm: {avg_weight_norm:.6f}")
    print(f"Total active neurons: {total_active}")
    print(f"Average CE Loss: {avg_ce_loss.item():.6f}")
    print(f"Average Pruning Loss: {avg_pruning_loss.item():.6f}")
    print(f"Average Loss: {avg_loss.item():.6f}")
    print(f"Total datapoints processed: {current_datapoints}")

    # Return metrics at the end of the segment
    return beta, total_samples, current_datapoints, active_neurons, accuracy


def test(model, device, test_loader):
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


def print_network_statistics(model):
    """Print statistics about the network's pruned state and return total active neurons"""
    print("\nNeurons Active:")
    total_neurons = 0
    active_neurons = 0

    for i, layer in enumerate(model.fc_layers):
        layer_neurons = layer.weight.shape[0]
        active_layer_neurons = int(model.masks[i].sum().item())
        total_neurons += layer_neurons
        active_neurons += active_layer_neurons
        print(
            f"Layer {i+1}: {active_layer_neurons}/{layer_neurons} neurons active "
            f"({100.0 * active_layer_neurons / layer_neurons:.2f}%)"
        )

    # # Add input and output layer neurons (always active)
    # active_neurons += 794
    # total_neurons += 794

    print(
        f"\nTotal: {active_neurons}/{total_neurons} neurons active "
        f"({100.0 * active_neurons / total_neurons:.2f}%)"
    )
    return active_neurons


def save_accuracy_data(model_accuracies, filename="data/pruning.csv"):
    """Saves the accuracy data to a CSV file."""
    os.makedirs("data", exist_ok=True)
    csv_path = filename

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as csvfile:
        fieldnames = [
            "Active Neurons",
            "Beta",
            "Accuracy",
            "Epoch",
            "Datapoints",
            "Pruning Penalty",
            "Target Neurons",
            "Learning Rate",
            "Success",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for entry in model_accuracies:
            writer.writerow(entry)

    # Update the message to show singular/plural based on number of entries
    entry_count = len(model_accuracies)
    if entry_count == 1:
        print(f"Saved 1 result to {csv_path}")
    else:
        print(f"Saved {entry_count} results to {csv_path}")


def filter_even_digits(dataset):
    """Filter the dataset to include only even digits."""
    even_indices = [
        i for i, (_, target) in enumerate(dataset) if target in {0, 2, 4, 6, 8}
    ]
    return torch.utils.data.Subset(dataset, even_indices)


def print_final_summary(all_results, target_neurons_list):
    """Print a comprehensive summary of all results after all sweeps are complete"""
    print(f"\n\n{'='*120}")
    print(f"FINAL SUMMARY OF ALL RESULTS")
    print(f"{'='*120}")

    # Group results by target neurons
    results_by_target = {}
    for target in target_neurons_list:
        results_by_target[target] = [
            r for r in all_results if r["Target Neurons"] == target
        ]

    # Print header
    print(
        f"{'Target':<10} {'Beta':<10} {'LR':<10} {'Active':<10} {'Accuracy':<10} {'Epochs':<10} {'Datapoints':<12} {'Success':<8}"
    )
    print(f"{'-'*120}")

    # For each target, print the best configuration
    for target in target_neurons_list:
        target_results = results_by_target[target]

        # Find the best result for this target
        best_result = None
        for result in target_results:
            if best_result is None:
                best_result = result
                continue

            # Update best result if current is better
            is_better = False
            if result["Success"] and not best_result["Success"]:
                is_better = True
            elif result["Success"] == best_result["Success"]:
                if result["Accuracy"] > best_result["Accuracy"]:
                    is_better = True
                elif (
                    result["Accuracy"] == best_result["Accuracy"]
                    and result["Active Neurons"] < best_result["Active Neurons"]
                ):
                    is_better = True
                elif (
                    result["Accuracy"] == best_result["Accuracy"]
                    and result["Active Neurons"] == best_result["Active Neurons"]
                    and result["Datapoints"] < best_result["Datapoints"]
                ):
                    is_better = True

            if is_better:
                best_result = result

        # Print the best result for this target
        if best_result:
            print(
                f"{target:<10} {best_result['Beta']:<10.3e} {best_result['Learning Rate']:<10.3e} "
                f"{best_result['Active Neurons']:<10} {best_result['Accuracy']:<10.2f} "
                f"{best_result['Epoch']:<10} {best_result['Datapoints']:<12} "
                f"{'✓' if best_result['Success'] else '✗'}"
            )

    print(f"{'-'*120}")

    # Print global statistics
    successful_configs = [r for r in all_results if r["Success"]]
    success_rate = len(successful_configs) / len(all_results) if all_results else 0

    print(f"Total configurations tested: {len(all_results)}")
    print(
        f"Successful configurations: {len(successful_configs)} ({success_rate*100:.1f}%)"
    )

    if successful_configs:
        min_neurons = min(r["Active Neurons"] for r in successful_configs)
        best_accuracy = max(r["Accuracy"] for r in successful_configs)
        print(f"Minimum active neurons in a successful configuration: {min_neurons}")
        print(f"Best accuracy in a successful configuration: {best_accuracy:.2f}%")

    print(f"{'='*120}")


def save_checkpoint_data(checkpoint_data, filename="data/pruning_checkpoints.json"):
    """Saves the intermediate checkpoint data from the best run to a JSON file."""
    os.makedirs("data", exist_ok=True)

    # Create structure for new run
    run_id = 1

    # Load existing data if file exists
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        try:
            with open(filename, "r") as f:
                all_runs = json.load(f)
                if all_runs and isinstance(all_runs, list):
                    run_id = len(all_runs) + 1
        except json.JSONDecodeError:
            # If file exists but is not valid JSON, start fresh
            all_runs = []
    else:
        all_runs = []

    # Prepare run data
    run_data = {
        "run_id": run_id,
        "metadata": {
            "target_neurons": checkpoint_data["metadata"]["target_neurons"],
            "beta": checkpoint_data["metadata"]["beta"],
            "lr": checkpoint_data["metadata"]["lr"],
            "penalty_type": checkpoint_data["metadata"]["penalty_type"],
            "final_active_neurons": checkpoint_data["metadata"]["final_active_neurons"],
            "final_accuracy": checkpoint_data["metadata"]["final_accuracy"],
            "final_datapoints": checkpoint_data["metadata"]["final_datapoints"],
            "success": checkpoint_data["metadata"]["success"],
        },
        "checkpoints": [],
    }

    # Add checkpoint data
    for cp in checkpoint_data["checkpoints"]:
        run_data["checkpoints"].append(
            {
                "epoch": cp["epoch"],
                "segment": cp["segment"],
                "datapoints": cp["datapoints"],
                "active_neurons": cp["active_neurons"],
                "accuracy": cp["accuracy"],
                "beta": cp["beta"],
            }
        )

    # Add to all runs and save
    all_runs.append(run_data)
    with open(filename, "w") as f:
        json.dump(all_runs, f, indent=2)

    print(
        f"Saved run #{run_id} with {len(checkpoint_data['checkpoints'])} checkpoints to {filename}"
    )


# ===========================
# Main Execution Loop
# ===========================


def main():
    # Hyperparameters
    even_digits_only = True
    pruning_penalty = "tied_l2"
    hidden_dim = 1200
    batch_size = 16
    target_datapoints = 50000
    pruning_threshold = 0.05
    num_runs = 10  # Number of runs for each hyperparameter configuration

    # List of target neuron counts to evaluate
    target_neurons_list = [600, 400, 240, 100, 70]  # add back in 28
    # target_neurons_list = [35, 25]

    # Fixed accuracy target
    target_accuracy = 97.0  # Target accuracy (%)
    max_datapoints = 50000  # Maximum datapoints to process before giving up

    # Expanded sweep ranges for more thorough exploration
    beta_values = [1e-3, 3e-3, 4e-3, 5e-3, 6.5e-3, 8e-3]  # add a higher option
    # beta_values = [1e-3, 5e-3, 1e-2]
    learning_rates = [
        4e-4,
        8e-4,
        1e-3,
        2e-3,
        3e-3,
        5e-3,
        8e-3,
    ]  # add back in 1e-2 to go really low
    # learning_rates = [5e-3, 8e-3, 1e-2]

    # Focused sweep around 1e-2, 1e-2 (staying within 5e-3 to 5e-1)
    # 1.5e-2, 8e-3 works rly well for n=100 neurons remaining to chop down a
    # ton of the network
    # beta_values = [1.5e-2]
    # learning_rates = [8e-3]

    # Enable 1/16-epoch checking
    check_partial_epochs = True

    # Data loading & preprocessing
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Filter datasets to include only even digits
    if even_digits_only:
        train_dataset = filter_even_digits(train_dataset)
        test_dataset = filter_even_digits(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the original model
    original_model_path = "models/original_model.pth"
    if not os.path.exists(original_model_path):
        raise FileNotFoundError(
            "Please train the original model first using train_original.py"
        )

    # Store all results
    all_results = []
    best_configs_by_target = {}  # To track best config for each target

    # Loop through each target neuron count
    for target_neurons in target_neurons_list:
        print(f"\n{'='*80}")
        print(f"STARTING SWEEP FOR TARGET NEURONS: {target_neurons}")
        print(f"{'='*80}")

        sweep_results = []  # Store results for this target neuron count

        # Track best hyperparameters and results for this target
        best_config = {
            "beta": None,
            "lr": None,
            "accuracy": 0.0,
            "active_neurons": float("inf"),
            "epochs": 0,
            "datapoints": float("inf"),
            "success": False,
            "target_neurons": target_neurons,
        }

        # Store checkpoints for the best run
        best_run_checkpoints = None

        # Print sweep information
        print(f"Starting hyperparameter sweep with:")
        print(f"Learning rates: {learning_rates}")
        print(f"Beta values: {beta_values}")
        print(f"Target neurons: {target_neurons}, Target accuracy: {target_accuracy}%")
        print(f"Number of runs per configuration: {num_runs}")

        for lr in learning_rates:
            for initial_beta in beta_values:
                print(f"\n{'-'*80}")
                print(
                    f"Pruning with Target neurons = {target_neurons}, Beta = {initial_beta}, Learning Rate = {lr}"
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

                    # Initialize prunable model and load original weights
                    model = Net(
                        hidden_dims=[hidden_dim, hidden_dim],
                        pruning_threshold=pruning_threshold,
                    ).to(device)
                    model.load_state_dict(torch.load(original_model_path))
                    original_accuracy = test(model, device, test_loader)
                    print(f"Original Model Accuracy: {original_accuracy:.2f}%")

                    # Use Adam optimizer instead of SGD
                    optimizer = optim.Adam(model.parameters(), lr=lr)

                    # Calculate T_max for scheduler, ensuring it's at least 1
                    dataset_size = len(train_loader.dataset)
                    t_max = max(1, target_datapoints // dataset_size)
                    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=lr)

                    total_datapoints = 0
                    epoch = 0

                    # Current beta value - can change during training
                    beta = initial_beta

                    # Variables to track our targets
                    accuracy_target_reached = False
                    neurons_target_reached = False
                    total_active_neurons = float("inf")
                    current_accuracy = 0.0

                    # Track checkpoints for this run
                    checkpoints = []

                    # Train until we've reached both targets or hit max datapoints
                    while (
                        not (accuracy_target_reached and neurons_target_reached)
                        and total_datapoints < max_datapoints
                    ):
                        epoch += 1
                        scheduler.step(epoch)

                        # Go through each 1/16 of the epoch
                        for segment in range(16):
                            if not check_partial_epochs and segment > 0:
                                continue  # Skip segments 1-15 if not checking partial epochs

                            # Train on this segment of the data
                            (
                                beta,  # Use the returned beta value instead of ignoring it
                                batch_samples,
                                current_datapoints,
                                active_neurons,
                                train_accuracy,
                            ) = train(
                                model,
                                device,
                                train_loader,
                                optimizer,
                                epoch,
                                beta=beta,
                                penalty_type=pruning_penalty,
                                total_datapoints=total_datapoints,
                                partial_epoch=check_partial_epochs,
                                segment_index=segment,
                            )

                            total_datapoints = current_datapoints
                            total_active_neurons = sum(active_neurons)

                            # Test after this segment
                            current_accuracy = test(model, device, test_loader)

                            # Save checkpoint data
                            checkpoints.append(
                                {
                                    "epoch": epoch,
                                    "segment": f"{segment+1}/16",
                                    "datapoints": total_datapoints,
                                    "active_neurons": total_active_neurons,
                                    "accuracy": current_accuracy,
                                    "beta": beta,
                                }
                            )

                            # Check if we've reached the neuron target (and stop pruning if so)
                            if total_active_neurons <= target_neurons:
                                neurons_target_reached = True
                                if beta > 0:
                                    print(
                                        f"Reached neuron target of {target_neurons} (current: {total_active_neurons}). Stopping pruning."
                                    )
                                    beta = 0.0
                            else:
                                neurons_target_reached = False

                            # Check if we've reached the accuracy target
                            accuracy_target_reached = (
                                current_accuracy >= target_accuracy
                            )

                            # Check if we've reached both targets simultaneously
                            if neurons_target_reached and accuracy_target_reached:
                                print(
                                    f"SUCCESS! Reached both targets after segment {segment+1} of epoch {epoch}: "
                                    f"{total_active_neurons} neurons with {current_accuracy:.2f}% accuracy"
                                )
                                break  # Exit the segment loop

                        # If both targets are met, exit the epoch loop too
                        if neurons_target_reached and accuracy_target_reached:
                            break

                    # Final status message for this run
                    success = neurons_target_reached and accuracy_target_reached
                    if success:
                        print(
                            f"Run {run+1}: Successfully reached both targets after {epoch} epochs and {total_datapoints} datapoints"
                        )
                    else:
                        print(
                            f"Run {run+1}: Failed to reach all targets after {epoch} epochs and {total_datapoints} datapoints"
                        )
                        print(
                            f"Final state: {total_active_neurons} neurons with {current_accuracy:.2f}% accuracy"
                        )
                        if neurons_target_reached:
                            print(f"Neuron target was reached")
                        if accuracy_target_reached:
                            print(f"Accuracy target was reached")

                    # Store results for this run
                    run_active_neurons.append(total_active_neurons)
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
                print(f"SUMMARY OF {num_runs} RUNS FOR Beta={initial_beta}, LR={lr}")
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
                    "Beta": initial_beta,
                    "Accuracy": avg_accuracy,
                    "Epoch": avg_epochs,
                    "Datapoints": avg_datapoints,
                    "Pruning Penalty": pruning_penalty,
                    "Target Neurons": target_neurons,
                    "Learning Rate": lr,
                    "Success": success_rate
                    >= 0.5,  # Success if at least half the runs succeeded
                }

                # Add to sweep results
                sweep_results.append(result)

                # Add to results collection
                all_results.append(result)

                # Check if this single run is better than the best config's current average
                is_better_run = False
                if success and (
                    not best_config["success"]
                    or (success and total_datapoints < best_config["datapoints"])
                    or (
                        success
                        and total_datapoints == best_config["datapoints"]
                        and current_accuracy > best_config["accuracy"]
                    )
                ):
                    is_better_run = True
                    # Save checkpoints for this run as it's better than previous best run
                    best_run_checkpoints = {
                        "metadata": {
                            "target_neurons": target_neurons,
                            "beta": initial_beta,
                            "lr": lr,
                            "penalty_type": pruning_penalty,
                            "final_active_neurons": total_active_neurons,
                            "final_accuracy": current_accuracy,
                            "final_datapoints": total_datapoints,
                            "success": success,
                        },
                        "checkpoints": checkpoints,
                    }

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
                    is_better or best_config["beta"] is None
                ):  # Also update if this is the first run
                    best_config["beta"] = initial_beta
                    best_config["lr"] = lr
                    best_config["accuracy"] = avg_accuracy
                    best_config["active_neurons"] = avg_active_neurons
                    best_config["epochs"] = avg_epochs
                    best_config["datapoints"] = avg_datapoints
                    best_config["success"] = result["Success"]

        # Store the best configuration for this target neuron count
        best_configs_by_target[target_neurons] = best_config

        # Save the best configuration for this target neuron count to CSV
        if best_config["success"]:
            best_result = {
                "Active Neurons": int(best_config["active_neurons"]),
                "Beta": best_config["beta"],
                "Accuracy": best_config["accuracy"],
                "Epoch": best_config["epochs"],
                "Datapoints": best_config["datapoints"],
                "Pruning Penalty": pruning_penalty,
                "Target Neurons": target_neurons,
                "Learning Rate": best_config["lr"],
                "Success": best_config["success"],
            }
            save_accuracy_data([best_result])
            print(f"\nSaved best configuration to data/pruning.csv")
        else:
            print(
                f"\nNo successful configuration found for target neurons {target_neurons}"
            )

        # Print summary table for this target neuron count
        print(f"\n{'-'*100}")
        print(f"SUMMARY FOR TARGET NEURONS: {target_neurons}")
        print(f"{'-'*100}")
        print(
            f"{'Beta':<10} {'LR':<10} {'Active':<10} {'Accuracy':<10} {'Epochs':<10} {'Datapoints':<12} {'Success':<8}"
        )
        print(f"{'-'*100}")

        for result in sweep_results:
            print(
                f"{result['Beta']:<10.3e} {result['Learning Rate']:<10.3e} "
                f"{result['Active Neurons']:<10} {result['Accuracy']:<10.2f} "
                f"{result['Epoch']:<10.1f} {result['Datapoints']:<12.1f} "
                f"{'✓' if result['Success'] else '✗'}"
            )

        print(f"{'-'*100}")
        print("BEST CONFIGURATION:")
        if best_config["beta"] is not None:
            print(f"Beta: {best_config['beta']:.3e}, LR: {best_config['lr']:.3e}")
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

        # Save checkpoint data for the best run if it was successful
        if best_run_checkpoints is not None and best_config["success"]:
            save_checkpoint_data(best_run_checkpoints)

    # Print final summary of best configurations for each target
    print(f"\n{'='*100}")
    print(f"BEST CONFIGURATIONS SUMMARY")
    print(f"{'='*100}")
    print(
        f"{'Target':<10} {'Beta':<10} {'LR':<10} {'Active':<10} {'Accuracy':<10} {'Datapoints':<12} {'Success':<8}"
    )
    print(f"{'-'*100}")

    for target, config in best_configs_by_target.items():
        if config["beta"] is not None:
            print(
                f"{target:<10} {config['beta']:<10.3e} {config['lr']:<10.3e} "
                f"{config['active_neurons']:<10.1f} {config['accuracy']:<10.2f} "
                f"{config['datapoints']:<12.1f} {'✓' if config['success'] else '✗'}"
            )
        else:
            print(
                f"{target:<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<12} ✗"
            )

    print(f"{'-'*100}")

    # Print final summary table with all results
    print_final_summary(all_results, target_neurons_list)


if __name__ == "__main__":
    main()
