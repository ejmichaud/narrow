import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import csv
from datetime import datetime
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
alpha = 0.1  # alpha is the weight for the cross-entropy loss, 1 - alpha is the weight for the distillation loss
temperatures = [20]  # 1, 5, 10
num_epochs = 8
hidden_dim_teacher = 1200  # Hidden dimension for the teacher model

# Hidden dimensions to test for students
hidden_dims = [
    15,
    25,
    35,
    45,
    50,
    65,
    80,
    100,
    120,
    140,
    200,
    300,
    400,
    500,
    600,
    800,
    1000,
    1200,
]

# Create transform for images into tensors
transform = transforms.ToTensor()

# Set up MNIST train and test sets for the teacher model
teacher_train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
teacher_test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)


def filter_even_digits(dataset):
    """
    Filters the dataset to only include samples with even digit labels.

    Args:
        dataset: The dataset to filter.

    Returns:
        A filtered dataset containing only even digit samples.
    """
    indices = [i for i, (_, label) in enumerate(dataset) if label % 2 == 0]
    return torch.utils.data.Subset(dataset, indices)


# Set up MNIST train and test sets for the student model
student_train_dataset = filter_even_digits(teacher_train_dataset)
student_test_dataset = filter_even_digits(teacher_test_dataset)

# Data loaders for teacher model
teacher_train_loader = DataLoader(
    teacher_train_dataset, batch_size=batch_size, shuffle=True
)
teacher_test_loader = DataLoader(
    teacher_test_dataset, batch_size=batch_size, shuffle=False
)

# Data loaders for student model
student_train_loader = DataLoader(
    student_train_dataset, batch_size=batch_size, shuffle=True
)
student_test_loader = DataLoader(
    student_test_dataset, batch_size=batch_size, shuffle=False
)


# Teacher model definition
class TeacherNet(nn.Module):
    def __init__(self, hidden_dim):
        super(TeacherNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Student model definition
class StudentNet(nn.Module):
    def __init__(self, hidden_dim):
        super(StudentNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Distillation loss function
def distillation_loss(y_student, y_teacher, T):
    """
    Computes the distillation loss between student and teacher outputs.

    Args:
        y_student: Logits from student model.
        y_teacher: Logits from teacher model.
        T: Temperature.

    Returns:
        Loss value.
    """
    p_student = F.log_softmax(y_student / T, dim=1)
    p_teacher = F.softmax(y_teacher / T, dim=1)
    # compensate for temperature scaling by multiplying by T^2
    loss = F.kl_div(p_student, p_teacher, reduction="batchmean") * (T * T)
    return loss


# Training function for teacher model
def train_teacher(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0
    datapoints_processed = epoch * len(train_loader.dataset)

    for batch_idx, (data, target) in enumerate(train_loader):
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

    total_loss /= total_samples
    accuracy = 100.0 * correct / total_samples
    print(
        "Teacher Train Epoch: {} \tDatapoints: {} \tLoss: {:.6f}\tAccuracy: {:.2f}%".format(
            epoch, datapoints_processed, total_loss, accuracy
        )
    )
    return accuracy


# Testing function
def test(model, device, test_loader, model_name="Model"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
    accuracy = 100.0 * correct / total
    print(f"\nTest set: {model_name} Accuracy: {correct}/{total} ({accuracy:.2f}%)\n")
    return accuracy


# Training function for student model with more frequent evaluation
def train_student_with_early_stopping(
    student_model,
    teacher_model,
    device,
    train_loader,
    test_loader,
    optimizer,
    T,
    alpha,
    target_accuracy=97.0,
):
    student_model.train()
    teacher_model.eval()

    total_datapoints = 0
    check_interval = (
        len(train_loader.dataset) // 8
    )  # Check accuracy every eighth of an epoch (changed from 4 to 8)

    # Track batches for reporting
    batch_loss_sum = 0
    batch_correct = 0
    batch_total = 0

    while True:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Compute student output
            student_logits = student_model(data)
            # Compute teacher output
            with torch.no_grad():
                teacher_logits = teacher_model(data)

            # Compute + combine losses
            loss_ce = F.cross_entropy(student_logits, target)
            loss_kd = distillation_loss(student_logits, teacher_logits, T)
            loss = alpha * loss_ce + (1.0 - alpha) * loss_kd

            loss.backward()
            optimizer.step()

            # Update batch statistics
            batch_loss_sum += loss.item() * data.size(0)
            pred = student_logits.argmax(dim=1, keepdim=True)
            batch_correct += pred.eq(target.view_as(pred)).sum().item()
            batch_total += data.size(0)

            # Update total datapoints processed
            total_datapoints += data.size(0)

            # Check accuracy periodically
            if total_datapoints % check_interval < batch_size:
                # Report training progress
                if batch_total > 0:
                    batch_loss = batch_loss_sum / batch_total
                    batch_accuracy = 100.0 * batch_correct / batch_total
                    print(
                        f"Student Training: Datapoints: {total_datapoints} \tLoss: {batch_loss:.6f}\tAccuracy: {batch_accuracy:.2f}%"
                    )
                    # Reset batch statistics
                    batch_loss_sum = 0
                    batch_correct = 0
                    batch_total = 0

                # Test current model
                student_acc = test(
                    student_model,
                    device,
                    test_loader,
                    model_name=f"Student (T={T})",
                )

                # Check if target accuracy reached
                if student_acc >= target_accuracy:
                    return total_datapoints, student_acc

                # Set back to training mode
                student_model.train()

    # This should not be reached with the infinite loop, but just in case
    return total_datapoints, student_acc


# Save training data
def save_training_data(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="") as csvfile:
        fieldnames = [
            "Epoch",
            "Datapoints",
            "Neurons Remaining",
            "Accuracy",
            "Temperature",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data:
            writer.writerow(entry)


# Main script
def main():
    student_results = []
    num_runs = 50  # Number of training runs for each configuration

    # Train the teacher model
    teacher_model = TeacherNet(hidden_dim_teacher).to(device)
    optimizer_teacher = optim.SGD(teacher_model.parameters(), lr=0.05, momentum=0.9)
    print("Training Teacher Model...")
    for epoch in range(1, num_epochs + 1):
        train_teacher(
            teacher_model, device, teacher_train_loader, optimizer_teacher, epoch
        )
    teacher_acc = test(teacher_model, device, teacher_test_loader, model_name="Teacher")

    for hidden_dim in hidden_dims:
        for T in temperatures:
            print(
                f"\nTraining {num_runs} students with hidden dimension: {hidden_dim} and temperature T={T}"
            )

            # Lists to store metrics for averaging
            run_datapoints = []
            run_accuracies = []

            for run in range(1, num_runs + 1):
                print(f"Run {run}/{num_runs}")
                student_model = StudentNet(hidden_dim).to(device)
                optimizer_student = optim.SGD(
                    student_model.parameters(), lr=0.01, momentum=0.9
                )

                # Train with early stopping
                datapoints_processed, student_acc = train_student_with_early_stopping(
                    student_model,
                    teacher_model,
                    device,
                    student_train_loader,
                    student_test_loader,
                    optimizer_student,
                    T,
                    alpha,
                    target_accuracy=97.0,
                )

                # Store metrics for this run
                run_datapoints.append(datapoints_processed)
                run_accuracies.append(student_acc)

            # Calculate averages
            avg_datapoints = sum(run_datapoints) / num_runs
            avg_accuracy = sum(run_accuracies) / num_runs

            print(f"Average results for hidden_dim={hidden_dim}, T={T}:")
            print(f"  Datapoints: {avg_datapoints:.2f}")
            print(f"  Accuracy: {avg_accuracy:.2f}%")

            # Append averaged results
            student_results.append(
                {
                    "Epoch": avg_datapoints / len(student_train_loader.dataset),
                    "Datapoints": avg_datapoints,
                    "Neurons Remaining": 2 * hidden_dim,
                    "Accuracy": avg_accuracy,
                    "Temperature": T,
                }
            )

    # Save results to a CSV file
    save_training_data(student_results, "data/student_training_accuracies.csv")
    print(
        f"Saved {len(student_results)} entries to {os.path.abspath('data/student_training_accuracies.csv')}"
    )


if __name__ == "__main__":
    main()
