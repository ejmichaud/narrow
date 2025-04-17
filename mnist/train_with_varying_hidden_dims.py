import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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


def train(model, device, train_loader, optimizer, epoch, segment, datapoints_seen):
    model.train()
    correct = 0
    total_samples = 0

    # Calculate start and end indices for this segment of the epoch
    total_batches = len(train_loader)
    segment_size = total_batches // 8
    start_idx = segment * segment_size
    end_idx = (segment + 1) * segment_size if segment < 7 else total_batches

    for i, (data, target) in enumerate(train_loader):
        # Skip batches not in this segment
        if i < start_idx or i >= end_idx:
            continue

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += data.size(0)
        datapoints_seen += data.size(0)

    accuracy = 100.0 * correct / total_samples if total_samples > 0 else 0
    epoch_str = f"{epoch}.{segment+1}"
    print(f"Epoch {epoch_str}: Training accuracy: {accuracy:.2f}%")
    return accuracy, datapoints_seen


def test(model, device, test_loader, epoch, segment):
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
    epoch_str = f"{epoch}.{segment+1}"
    print(f"Epoch {epoch_str}: Test accuracy: {accuracy:.2f}%")
    return accuracy


def save_training_data(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["Epoch", "Datapoints", "Neurons Remaining", "Accuracy"]
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

    hidden_dims_list = [
        [15],
        [25],
        [50],
        [80],
        [140],
        [200],
        [300],
        [400],
        [500],
        [600],
        [700],
        [800],
        [900],
        [1000],
        [1100],
        [1200],
    ]
    num_epochs = 20
    num_runs = 50  # Number of networks to train for each hidden dimension

    all_training_data = []

    for hidden_dims in hidden_dims_list:
        print(f"Training for hidden dimension: {hidden_dims[0]}")

        # Lists to collect statistics across runs
        run_epochs = []
        run_datapoints = []
        run_neurons = []
        run_accuracies = []

        for run in range(num_runs):
            print(f"Run {run+1}/{num_runs}")
            model = Net(hidden_dim=hidden_dims[0]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

            datapoints_seen = 0
            reached_target = False
            final_test_accuracy = 0
            final_epoch = 0
            final_segment = 0

            for epoch in range(1, num_epochs + 1):
                for segment in range(8):  # 0-7 for eight segments per epoch
                    train_accuracy, datapoints_seen = train(
                        model,
                        device,
                        train_loader,
                        optimizer,
                        epoch,
                        segment,
                        datapoints_seen,
                    )
                    test_accuracy = test(model, device, test_loader, epoch, segment)

                    final_test_accuracy = test_accuracy
                    final_epoch = epoch
                    final_segment = segment

                    if test_accuracy >= 97.0:
                        reached_target = True
                        break

                if reached_target:
                    break

            # Collect statistics for this run
            neurons_remaining = sum(
                [layer.weight.shape[0] for layer in model.fc_layers]
            )

            run_epochs.append(f"{final_epoch}.{final_segment+1}")
            run_datapoints.append(datapoints_seen)
            run_neurons.append(neurons_remaining)
            run_accuracies.append(final_test_accuracy)

        # Calculate averages across all runs
        avg_epoch = sum([float(e) for e in run_epochs]) / num_runs
        avg_datapoints = sum(run_datapoints) / num_runs
        avg_neurons = sum(run_neurons) / num_runs
        avg_accuracy = sum(run_accuracies) / num_runs

        # Record the average data for this hidden dimension
        all_training_data.append(
            {
                "Epoch": f"{avg_epoch:.2f}",
                "Datapoints": int(avg_datapoints),
                "Neurons Remaining": int(avg_neurons),
                "Accuracy": avg_accuracy,
            }
        )

    save_training_data(all_training_data, "data/training_accuracies_updated.csv")
    print(
        f"Saved {len(all_training_data)} entries to {os.path.abspath('data/training_accuracies_updated.csv')}"
    )


if __name__ == "__main__":
    main()
