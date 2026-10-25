"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load entire test set
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test_metrics(net, testloader, device):
    """Evaluate the model and return classification and confidence metrics."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    total_examples = 0
    confidence_sum = 0.0
    correct_confidence_sum = 0.0
    wrong_confidence_sum = 0.0
    # Accumulate class counts for macro-averaged metrics.
    true_positive = torch.zeros(10, dtype=torch.long, device=device)
    predicted_counts = torch.zeros(10, dtype=torch.long, device=device)
    actual_counts = torch.zeros(10, dtype=torch.long, device=device)
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predictions = probabilities.max(dim=1)
            matches = predictions == labels
            loss += criterion(outputs, labels).item()
            correct += matches.sum().item()
            total_examples += labels.size(0)
            # Track average confidence for all, correct, and incorrect predictions.
            confidence_sum += confidence.sum().item()
            correct_confidence_sum += confidence[matches].sum().item()
            wrong_confidence_sum += confidence[~matches].sum().item()
            true_positive += torch.bincount(labels[matches], minlength=10)
            predicted_counts += torch.bincount(predictions, minlength=10)
            actual_counts += torch.bincount(labels, minlength=10)
    # Preserve the original loss and accuracy calculation.
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    # Calculate precision, recall, and F1 for each class.
    precision_per_class = true_positive.float() / predicted_counts.clamp(min=1)
    recall_per_class = true_positive.float() / actual_counts.clamp(min=1)
    f1_per_class = (
        2 * precision_per_class * recall_per_class
        / (precision_per_class + recall_per_class).clamp(min=1e-12)
    )
    # Return macro-averaged classification metrics and mean confidence.
    return {
        "eval_loss": loss,
        "eval_acc": accuracy,
        "eval_precision": precision_per_class.mean().item(),
        "eval_recall": recall_per_class.mean().item(),
        "eval_f1": f1_per_class.mean().item(),
        "eval_confidence": confidence_sum / max(total_examples, 1),
        "eval_correct_confidence": correct_confidence_sum / max(correct, 1),
        "eval_wrong_confidence": wrong_confidence_sum / max(total_examples - correct, 1),
    }


def test(net, testloader, device):
    """Validate the model and return loss and accuracy."""
    metrics = test_metrics(net, testloader, device)
    return metrics["eval_loss"], metrics["eval_acc"]
