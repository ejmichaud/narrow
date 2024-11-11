# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import csv
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        """Update masks based on weight magnitudes relative to mean"""
        active_neurons = []

        for i, layer in enumerate(self.fc_layers):
            # Calculate L1 norm of incoming weights for each neuron
            weight_norms = torch.norm(layer.weight, p=1, dim=1)
            # Calculate mean L2 norms
            mean_norm = weight_norms.mean()
            # Update mask: neurons with norm below threshold * mean are masked
            self.masks[i] = (weight_norms > self.pruning_threshold * mean_norm).float()
            layer.weight.data *= self.masks[i].unsqueeze(1)
            active_neurons.append(int(self.masks[i].sum().item()))

        return active_neurons


# options: l1, tied_l2, tied_l2_with_l1
def pruning_loss(model, penalty_type="l1"):
    penalty = 0.0
    for i in range(len(model.fc_layers) - 1):
        W1 = model.fc_layers[i].weight
        W2 = model.fc_layers[i + 1].weight

        if penalty_type == "tied_l2":
            # Concatenate W1 with transposed W2 along dim=1
            combined = torch.cat([W1, W2.t()], dim=1)
            # Take norm of each row (where each row represents all weights connected to one neuron)
            penalty = penalty + torch.norm(combined, p=2, dim=1).sum()

        elif penalty_type == "tied_l2_with_l1":
            combined = torch.cat([W1, W2.t()], dim=1)
            l2 = torch.norm(combined, p=2, dim=1).sum()
            l1 = torch.norm(combined, p=1, dim=1).sum()
            penalty = penalty + l2 / (2 * l1 ** (0.2))

        # should maybe change the other things to not penalize the masked weights being higher
        elif penalty_type == "l1":
            if i == len(model.fc_layers) - 2:
                W1 = model.fc_layers[i + 1].weight
            penalty = penalty + torch.norm(W1, p=1, dim=1).sum()

    return penalty


# Training and Testing
def train(
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    beta,
    penalty_type="l1",
):
    if epoch % 2 == 0:
        print(
            f"\nStarting training with beta={beta}, penalty_type={penalty_type}, epoch={epoch}"
        )
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # Compute loss
        loss_ce = F.cross_entropy(output, target)
        penalty = pruning_loss(model, penalty_type=penalty_type)
        # print(penalty.grad_fn)
        loss = loss_ce + beta * penalty

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Calculate average weight norm across all layers for print debugging
        total_norm = 0
        param_count = 0
        for layer in model.fc_layers:
            layer_norm = torch.norm(layer.weight, p=1)
            total_norm += layer_norm
            param_count += 1
        avg_weight_norm = total_norm / param_count

        # Update masks
        if beta > 0:
            active_neurons = model.update_masks()
            total_active = sum(active_neurons) + 10  # Add output layer neurons

        total_loss += loss
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += data.size(0)

        # Debug prints
        # for i, layer in enumerate(model.fc_layers):
        #     if layer.weight.grad is None and i != 0:
        #         print(f"Layer {i} has no gradient")

    if epoch % 2 == 0:
        avg_loss = total_loss / total_samples
        accuracy = 100.0 * correct / total_samples
        print(
            f"Train Epoch: {epoch} [{total_samples}/{len(train_loader.dataset)} "
            f"({100. * total_samples / len(train_loader.dataset):.0f}%)]\t"
            f"Loss: {avg_loss:.6f}\tAccuracy: {accuracy:.2f}%"
        )
        print(f"Active neurons per layer: {active_neurons}")
        print(f"Average weight norm: {avg_weight_norm:.6f}")
        print(f"Total active neurons: {total_active}")
        print(f"Cross Entropy Loss: {loss_ce.item():.6f}")
        print(f"L1 Penalty Term: {beta*penalty.item():.6f}")
        print(f"Total Loss: {loss.item():.6f}")


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

    # Add output layer neurons (always active)
    active_neurons += 10
    total_neurons += 10

    print(
        f"\nTotal: {active_neurons}/{total_neurons} neurons active "
        f"({100.0 * active_neurons / total_neurons:.2f}%)"
    )
    return active_neurons


# ===========================
# Main Execution Loop
# ===========================


def main():
    # Hyperparameters
    pruning_penalty = "l1"
    hidden_dim = 1200
    batch_size = 64
    learning_rate = 1e-1
    num_epochs = 8
    pruning_threshold = 0.08

    # Beta values for pruning loss
    beta_values = [5e-7]

    model_save_dir = "models"

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model_accuracies = []

    for beta in beta_values:
        print(f"Pruning with Beta = {beta} and {pruning_penalty} Penalty")

        # Initialize prunable model and load original weights
        model = Net(
            hidden_dims=[hidden_dim, hidden_dim],
            pruning_threshold=pruning_threshold,
        ).to(device)
        model.load_state_dict(torch.load(original_model_path))

        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        for epoch in range(1, num_epochs + 1):
            train(
                model,
                device,
                train_loader,
                optimizer,
                epoch,
                beta=beta,
                penalty_type=pruning_penalty,
            )

            # Save progress every 2 epochs
            if epoch % 2 == 0:
                accuracy = test(model, device, test_loader)
                active_neurons = print_network_statistics(model)

                model_accuracies.append(
                    {
                        "Active Neurons": active_neurons,
                        "Beta": beta,
                        "Accuracy": accuracy,
                    }
                )

        # Save model weights at the end of training
        os.makedirs(model_save_dir, exist_ok=True)
        model_filename = f"pruned_model_hidden{hidden_dim}_beta{beta}.pth"
        model_path = os.path.join(model_save_dir, model_filename)
        torch.save(model.state_dict(), model_path)
        print(f"Saved Model weights to {model_path}")

    os.makedirs("data", exist_ok=True)
    csv_path = "data/pruning_accuracies.csv"
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["Active Neurons", "Beta", "Accuracy"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in model_accuracies:
            writer.writerow(entry)
    print(f"Saved accuracy data to {csv_path}")


if __name__ == "__main__":
    main()
