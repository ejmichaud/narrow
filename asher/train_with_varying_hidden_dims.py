import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR


# Define the model class
class Net(nn.Module):
    def __init__(self, input_dim=28 * 28, hidden_dim=1200, output_dim=10):
        super(Net, self).__init__()
        self.fc_layers = nn.ModuleList()
        previous_dim = input_dim
        for _ in range(2):
            self.fc_layers.append(nn.Linear(previous_dim, hidden_dim))
            previous_dim = hidden_dim
        self.output_layer = nn.Linear(previous_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        x = self.output_layer(x)
        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    total_samples = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += data.size(0)
    accuracy = 100.0 * correct / total_samples
    print(f"Epoch {epoch}: Training accuracy: {accuracy:.2f}%")
    return accuracy


def test(model, device, test_loader, epoch):
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
    print(f"Epoch {epoch}: Test accuracy: {accuracy:.2f}%")
    return accuracy


def save_training_data(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["Epoch", "Neurons Remaining", "Accuracy"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data:
            writer.writerow(entry)


def filter_even_digits(dataset):
    """Filter the dataset to include only even digits."""
    even_indices = [
        i for i, (_, target) in enumerate(dataset) if target in {0, 2, 4, 6, 8}
    ]
    return torch.utils.data.Subset(dataset, even_indices)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Filter datasets to include only even digits
    train_dataset = filter_even_digits(train_dataset)
    test_dataset = filter_even_digits(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    hidden_dims_list = [[5], [10], [20], [80], [200], [800], [1200]]
    num_epochs = 20
    learning_rate = 0.01

    all_training_data = []

    for hidden_dims in hidden_dims_list:
        model = Net(hidden_dim=hidden_dims[0]).to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=learning_rate
        )

        for epoch in range(1, num_epochs + 1):
            scheduler.step(epoch)
            train_accuracy = train(model, device, train_loader, optimizer, epoch)
            test_accuracy = test(model, device, test_loader, epoch)
            neurons_remaining = sum(
                [layer.weight.shape[0] for layer in model.fc_layers]
            )
            if test_accuracy >= 97.0:
                break
        all_training_data.append(
            {
                "Epoch": epoch,
                "Neurons Remaining": neurons_remaining,
                "Accuracy": test_accuracy,
            }
        )

    save_training_data(all_training_data, "data/training_accuracies_updated.csv")
    print(
        f"Saved {len(all_training_data)} entries to {os.path.abspath('data/training_accuracies_updated.csv')}"
    )


if __name__ == "__main__":
    main()
