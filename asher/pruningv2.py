"""Asher loading this in on 3/1/2025"""

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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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


# options: l1, tied_l1, tied_l2, tied_l2_with_l1, tied_l1_with_l1, tied_l2_with_l2, tied_l2_with_lhalf, tied_l1_with_lhalf
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
            penalty = penalty + (torch.norm(combined, p=2, dim=1).sum()) / (
                W1.numel() + W2.numel()
            )

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
):
    print(
        f"\nStarting training with beta={beta}, penalty_type={penalty_type}, epoch={epoch}"
    )
    model.train()
    total_loss = 0
    total_ce_loss = 0
    correct = 0
    total_samples = 0
    total_norm = 0

    for batch_idx, (data, target) in enumerate(train_loader, 1):
        # PLOTTING
        if batch_idx % 100 == 1:
            visualize_mlp(
                model=model, weight_threshold=0.0, color_range=[-0.3, 0.3]
            )  # HERE
            plt.title(f"Step {len(os.listdir('data/weight_vis'))}")
            plt.savefig(
                f"data/weight_vis/pruning_step_{len(os.listdir('data/weight_vis')):04d}.png"
            )
            plt.close()

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # Compute loss
        loss_ce = F.cross_entropy(output, target)
        penalty = pruning_loss(model, penalty_type=penalty_type)
        detached_penalty = torch.tensor(penalty).detach()
        # beta=0.18 works well -- 0.23 for methods that take inverses
        loss = loss_ce + beta * (penalty)

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

        # Update masks
        if beta > 0:
            active_neurons = model.update_masks()
            total_active = sum(
                active_neurons
            )  # No longer adding input and output layer neurons

        total_loss += loss
        total_ce_loss += loss_ce
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += data.size(0)

    # Calculate average weight norm across all layers for printing
    param_count = 0
    for layer in model.fc_layers + [model.output_layer]:
        layer_norm = torch.norm(layer.weight, p=1)
        total_norm += layer_norm
        param_count += layer.weight.numel()
    avg_weight_norm = total_norm / param_count
    avg_loss = total_loss / total_samples
    avg_ce_loss = total_ce_loss / total_samples
    avg_pruning_loss = avg_loss - avg_ce_loss
    accuracy = 100.0 * correct / total_samples
    print(
        f"Train Epoch: {epoch} [{total_samples}/{len(train_loader.dataset)} "
        f"({100. * total_samples / len(train_loader.dataset):.0f}%)]\t"
        f"Loss: {avg_loss:.6f}\tAccuracy: {accuracy:.2f}%"
    )
    print(f"Active neurons per layer: {active_neurons}")
    print(f"Average weight norm: {avg_weight_norm:.6f}")
    print(f"Total active neurons: {total_active}")
    print(f"Average CE Loss: {avg_ce_loss.item():.6f}")
    print(f"Average Pruning Loss: {avg_pruning_loss.item():.6f}")
    print(f"Average Loss: {avg_loss.item():.6f}")

    # Log metrics to W&B
    # wandb.log(
    #     {
    #         "epoch": epoch,
    #         "average_weight_norm": avg_weight_norm.item(),
    #         "pruning_loss": avg_pruning_loss.item(),
    #         "ce_loss": avg_ce_loss.item(),
    #         "active_neurons": total_active,
    #         "accuracy": accuracy,
    #     }
    # )

    return beta, total_samples


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


def save_accuracy_data(model_accuracies, pruning_penalty):
    """Saves the accuracy data to a CSV file with a unique run number."""
    os.makedirs(f"data/{pruning_penalty}", exist_ok=True)
    # Get list of existing CSV files in the specific directory
    existing_csv_files = [
        f
        for f in os.listdir(f"data/{pruning_penalty}")
        if f.startswith("pruning_accuracies_run") and f.endswith(".csv")
    ]
    # Determine the next run number
    if existing_csv_files:
        run_numbers = [
            int(f.split("_run")[1].split(".csv")[0]) for f in existing_csv_files
        ]
        next_run_number = max(run_numbers) + 1
    else:
        next_run_number = 1
    csv_path = f"data/{pruning_penalty}/pruning_accuracies_run{next_run_number}.csv"
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["Active Neurons", "Beta", "Accuracy", "Epoch", "Total Datapoints"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in model_accuracies:
            writer.writerow(entry)
    print(f"Saved accuracy data to {csv_path}")


def filter_even_digits(dataset):
    """Filter the dataset to include only even digits."""
    even_indices = [
        i for i, (_, target) in enumerate(dataset) if target in {0, 2, 4, 6, 8}
    ]
    return torch.utils.data.Subset(dataset, even_indices)


# ===========================
# Main Execution Loop
# ===========================


def main():
    # Hyperparameters
    even_digits_only = True
    pruning_penalty = "tied_l2_with_lhalf"
    hidden_dim = 1200
    batch_size = 64
    num_epochs = 4
    pruning_threshold = 0.3  # up from 0.2
    # Beta values for pruning loss -- 0.2 is good
    # For tied_l2_with_lhalf: approximately 0.5 / average_pruning_loss = 64
    # For tied_l2: approximately 3 / average_pruning_loss = 3000
    # For tied_l1: approximately 500
    # For tied_l1_with_lhalf: approximately 25
    beta_values = [64]
    learning_rates = [2e-1]
    save_model_weights = False
    model_save_dir = "models"

    # Initialize W&B
    # wandb.init(
    #     project="pruning-experiment",
    #     config={
    #         "pruning_penalty": pruning_penalty,
    #         "hidden_dim": hidden_dim,
    #         "batch_size": batch_size,
    #         "num_epochs": num_epochs,
    #         "pruning_threshold": pruning_threshold,
    #         "beta_values": beta_values,
    #         "learning_rates": learning_rates,
    #     },
    # )

    # Load the original model
    original_model_path = "models/original_model.pth"
    if not os.path.exists(original_model_path):
        raise FileNotFoundError(
            "Please train the original model first using train_original.py"
        )

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

    model_accuracies = []

    for lr in learning_rates:
        for beta in beta_values:
            # wandb.config.update(
            #     {"learning_rate": lr, "beta": beta}, allow_val_change=True
            # )
            print(f"Pruning with Beta = {beta} and {pruning_penalty} Penalty")
            # Initialize prunable model and load original weights
            model = Net(
                hidden_dims=[hidden_dim, hidden_dim],
                pruning_threshold=pruning_threshold,
            ).to(device)
            model.load_state_dict(torch.load(original_model_path))
            original_accuracy = test(model, device, test_loader)
            print(f"Original Model Accuracy: {original_accuracy:.2f}%")

            # Save initial model statistics before training
            active_neurons = print_network_statistics(model)
            model_accuracies.append(
                {
                    "Active Neurons": active_neurons,
                    "Beta": beta,
                    "Accuracy": original_accuracy,
                    "Epoch": 0,
                    "Total Datapoints": 0,
                }
            )

            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr)

            total_datapoints = 0
            for epoch in range(1, num_epochs + 1):
                scheduler.step(epoch)
                beta, total_samples = train(
                    model,
                    device,
                    train_loader,
                    optimizer,
                    epoch,
                    beta=beta,
                    penalty_type=pruning_penalty,
                )
                total_datapoints += total_samples
                # Save progress every epoch
                accuracy = test(model, device, test_loader)
                active_neurons = print_network_statistics(model)
                model_accuracies.append(
                    {
                        "Active Neurons": active_neurons,
                        "Beta": beta,
                        "Accuracy": accuracy,
                        "Epoch": epoch,
                        "Total Datapoints": total_datapoints,
                    }
                )
                # Log metrics to W&B
                # wandb.log(
                #     {
                #         "epoch": epoch,
                #         "accuracy": accuracy,
                #         "active_neurons": active_neurons,
                #         "beta": beta,
                #         "total_datapoints": total_datapoints,
                #     }
                # )

            # Optional: save model weights at the end of training
            if save_model_weights:
                os.makedirs(model_save_dir, exist_ok=True)
                model_filename = f"pruned_model_hidden{hidden_dim}_beta{beta}.pth"
                model_path = os.path.join(model_save_dir, model_filename)
                torch.save(model.state_dict(), model_path)
                print(f"Saved Model weights to {model_path}")

    # Save accuracy data to CSV
    save_accuracy_data(model_accuracies, pruning_penalty)
    # wandb.finish()


if __name__ == "__main__":
    main()
