import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import csv
from datetime import datetime

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
alpha = 0.1  # alpha is the weight for the cross-entropy loss, 1 - alpha is the weight for the distillation loss
temperatures = [1, 5, 10]
num_epochs = 5
hidden_dim_teacher = 1200  # Hidden dimension for the teacher model

# Hidden dimensions to test for students
hidden_dims = [20, 80, 140, 200, 400, 800, 1000]

# Create transform for images into tensors
transform = transforms.ToTensor()

# set up MNIST train and test sets and dataloaders
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


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
        "Teacher Train Epoch: {} \tLoss: {:.6f}\tAccuracy: {:.2f}%".format(
            epoch, total_loss, accuracy
        )
    )


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


# Training function for student model
def train_student(
    student_model, teacher_model, device, train_loader, optimizer, epoch, T, alpha
):
    student_model.train()
    teacher_model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0

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

        total_loss += loss.item() * data.size(0)
        pred = student_logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += data.size(0)

    total_loss /= total_samples
    accuracy = 100.0 * correct / total_samples
    print(
        "Student Train Epoch: {} \tLoss: {:.6f}\tAccuracy: {:.2f}%".format(
            epoch, total_loss, accuracy
        )
    )


# Main script
def main():
    student_results = []

    # Train the teacher model
    teacher_model = TeacherNet(hidden_dim_teacher).to(device)
    optimizer_teacher = optim.SGD(teacher_model.parameters(), lr=0.01, momentum=0.9)
    print("Training Teacher Model...")
    for epoch in range(1, num_epochs + 1):
        train_teacher(teacher_model, device, train_loader, optimizer_teacher, epoch)
    teacher_acc = test(teacher_model, device, test_loader, model_name="Teacher")

    for hidden_dim in hidden_dims:
        for T in temperatures:
            print(
                f"\nTraining student with hidden dimension: {hidden_dim} and temperature T={T}"
            )
            student_model = StudentNet(hidden_dim).to(device)
            optimizer_student = optim.SGD(
                student_model.parameters(), lr=0.01, momentum=0.9
            )

            for epoch in range(1, num_epochs + 1):
                train_student(
                    student_model,
                    teacher_model,
                    device,
                    train_loader,
                    optimizer_student,
                    epoch,
                    T,
                    alpha,
                )

            student_acc = test(
                student_model, device, test_loader, model_name=f"Student (T={T})"
            )

            student_results.append(
                {"hidden_dim": hidden_dim, "temperature": T, "accuracy": student_acc}
            )

    # Save results to a CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_accuracies_{timestamp}.csv"
    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["hidden_dim", "temperature", "accuracy"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(student_results)


if __name__ == "__main__":
    main()
