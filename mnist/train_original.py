import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


class Net(nn.Module):
    def __init__(
        self,
        input_dim=28 * 28,
        hidden_dims=[1200, 1200],
        output_dim=10,
        dropout_rate=0.2,
    ):
        super(Net, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        previous_dim = input_dim

        for h_dim in hidden_dims:
            self.fc_layers.append(nn.Linear(previous_dim, h_dim))
            previous_dim = h_dim

        self.output_layer = nn.Linear(previous_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        for fc in self.fc_layers:
            x = F.relu(fc(x))
            x = self.dropout(x)  # Apply dropout after activation
        x = self.output_layer(x)
        return x


def train(
    model, device, train_loader, optimizer, epoch, total_datapoints, print_interval=100
):
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
        current_datapoints = total_datapoints + total_samples

        if batch_idx % print_interval == 0:
            avg_loss = total_loss / total_samples
            accuracy = 100.0 * correct / total_samples
            print(
                f"Train Epoch: {epoch} [{total_samples}/{len(train_loader.dataset)} "
                f"({100. * total_samples / len(train_loader.dataset):.0f}%)]\t"
                f"Total Datapoints: {current_datapoints}\t"
                f"Loss: {avg_loss:.6f}\tAccuracy: {accuracy:.2f}%"
            )


def test(model, device, test_loader, total_datapoints):
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
    print(
        f"Test set: Total Datapoints: {total_datapoints}, Accuracy: {correct}/{total_samples} ({accuracy:.2f}%)"
    )
    return accuracy


def main():
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    weight_decay = 1e-4  # L2 regularization
    dropout_rate = 0.2  # Dropout rate
    num_epochs = 20
    print_interval = 100
    hidden_dims = [1200, 1200]
    even_digits_only = True

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

    # Create model directory
    os.makedirs("models", exist_ok=True)

    # Create and train model
    model = Net(hidden_dims=hidden_dims, dropout_rate=dropout_rate).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=2, factor=0.5
    )

    # Training loop
    total_datapoints = 0
    best_accuracy = 0
    for epoch in range(1, num_epochs + 1):
        train(
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            total_datapoints,
            print_interval,
        )
        total_datapoints += len(train_loader.dataset)
        accuracy = test(model, device, test_loader, total_datapoints)

        # Update learning rate based on validation performance
        scheduler.step(accuracy)

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f"models/best_model.pth")
            print(f"Saved new best model with accuracy: {accuracy:.2f}%")

        # Check if test accuracy threshold is reached
        if accuracy >= 99:
            print(f"Reached desired test accuracy of {accuracy:.2f}% at epoch {epoch}")
            print(f"Total number of datapoints processed: {total_datapoints}")
            break

    # Load best model for final evaluation
    model.load_state_dict(torch.load(f"models/best_model.pth"))
    final_accuracy = test(model, device, test_loader, total_datapoints)
    print(f"Final model accuracy: {final_accuracy:.2f}%")

    # Save model
    torch.save(model.state_dict(), f"models/original_model.pth")
    print(f"Saved original model to models/original_model.pth")


if __name__ == "__main__":
    main()
