import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, input_dim=28 * 28, hidden_dims=[1200, 1200], output_dim=10):
        super(Net, self).__init__()
        self.fc_layers = nn.ModuleList()
        previous_dim = input_dim

        for h_dim in hidden_dims:
            self.fc_layers.append(nn.Linear(previous_dim, h_dim))
            previous_dim = h_dim

        self.output_layer = nn.Linear(previous_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        x = self.output_layer(x)
        return x


def train(model, device, train_loader, optimizer, epoch, print_interval=100):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += data.size(0)

        if batch_idx % print_interval == 0:
            avg_loss = total_loss / total_samples
            accuracy = 100.0 * correct / total_samples
            print(
                f"Train Epoch: {epoch} [{total_samples}/{len(train_loader.dataset)} "
                f"({100. * total_samples / len(train_loader.dataset):.0f}%)]\t"
                f"Loss: {avg_loss:.6f}\tAccuracy: {accuracy:.2f}%"
            )


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


def main():
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.01
    num_epochs = 5
    print_interval = 100
    hidden_dims = [1200, 1200]

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

    # Create and train model
    model = Net(hidden_dims=hidden_dims).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, print_interval)

    # Final test
    test(model, device, test_loader)

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/original_model.pth")
    print("Saved original model to models/original_model.pth")


if __name__ == "__main__":
    main()
