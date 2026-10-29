"""UCI HAR task utilities for the Flower / PyTorch app."""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

NUM_FEATURES = 561
NUM_CLASSES = 6
DATASET_DIR = Path(__file__).resolve().parent.parent / "data" / "UCI HAR Dataset"
DATASET_FILES = tuple(
    f"{split}/{filename}_{split}.txt"
    for split in ("train", "test")
    for filename in ("X", "y", "subject")
)


class Net(nn.Module):
    """Multilayer perceptron for the 561 UCI HAR features."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_FEATURES, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


_dataset_cache: dict[tuple[str, str], TensorDataset] = {}


def _dataset_files_exist(dataset_dir: Path) -> bool:
    return all((dataset_dir / filename).is_file() for filename in DATASET_FILES)


def resolve_dataset_dir(configured_dir: str | Path | None = None) -> Path:
    """Resolve and validate the directory containing the local UCI HAR files."""
    configured_dir = configured_dir or os.environ.get("UCI_HAR_DATASET_DIR")
    if configured_dir:
        dataset_dir = Path(configured_dir).expanduser()
        if not dataset_dir.is_absolute():
            raise ValueError(
                f"dataset-dir must be an absolute path, got {str(dataset_dir)!r}. "
                "Pass the resolved data folder with --run-config."
            )
        if not _dataset_files_exist(dataset_dir):
            raise FileNotFoundError(
                f"UCI HAR files not found in {dataset_dir}. Check that it contains "
                "train/X_train.txt, train/y_train.txt, train/subject_train.txt "
                "and the corresponding test files."
            )
        return dataset_dir

    if _dataset_files_exist(DATASET_DIR):
        return DATASET_DIR

    raise FileNotFoundError(
        "UCI HAR dataset path is required when running from a Flower bundle. "
        "In PowerShell, set `$datasetDir = (Resolve-Path '.\\data\\UCI HAR "
        "Dataset').Path` and pass `--run-config \"dataset-dir='$datasetDir'\"`."
    )


def _load_dataset(
    split: str, dataset_dir: str | Path | None = None
) -> TensorDataset:
    """Read and cache one official UCI HAR split from the local dataset files."""
    if split not in ("train", "test"):
        raise ValueError(f"Unknown UCI HAR split: {split}")
    resolved_dir = resolve_dataset_dir(dataset_dir)
    cache_key = (split, str(resolved_dir.resolve()))
    if cache_key not in _dataset_cache:
        features = np.loadtxt(
            resolved_dir / split / f"X_{split}.txt", dtype=np.float32
        )
        labels = np.loadtxt(
            resolved_dir / split / f"y_{split}.txt", dtype=np.int64
        )
        subjects = np.loadtxt(
            resolved_dir / split / f"subject_{split}.txt", dtype=np.int64
        )
        labels = np.asarray(labels).reshape(-1) - 1
        subjects = np.asarray(subjects).reshape(-1)
        if features.ndim != 2 or features.shape[1] != NUM_FEATURES:
            raise ValueError(
                f"Expected {NUM_FEATURES} features in the UCI HAR {split} split, "
                f"got shape {features.shape}"
            )
        if (
            features.shape[0] != labels.shape[0]
            or features.shape[0] != subjects.shape[0]
        ):
            raise ValueError(
                f"UCI HAR {split} features, labels, and subjects have different "
                "row counts"
            )
        if labels.size and (labels.min() < 0 or labels.max() >= NUM_CLASSES):
            raise ValueError(f"UCI HAR {split} labels must be in the range 1-6")
        _dataset_cache[cache_key] = TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(labels.copy()),
            torch.from_numpy(subjects.copy()),
        )
    return _dataset_cache[cache_key]


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    dataset_dir: str | Path | None = None,
):
    """Load one participant's data and split it into local train/validation."""
    dataset = _load_dataset("train", dataset_dir)
    features, labels, subjects = dataset.tensors
    participant_ids = torch.unique(subjects, sorted=True)
    if num_partitions != len(participant_ids):
        raise ValueError(
            f"UCI HAR has {len(participant_ids)} training participants; configure "
            f"exactly {len(participant_ids)} Flower clients, got {num_partitions}"
        )
    if not 0 <= partition_id < num_partitions:
        raise ValueError(
            f"partition_id must be in [0, {num_partitions}), got {partition_id}"
        )

    participant_id = participant_ids[partition_id]
    partition_indices = torch.where(subjects == participant_id)[0]
    if len(partition_indices) < 2:
        raise ValueError(
            f"UCI HAR participant {participant_id.item()} needs at least two "
            "examples for training and validation"
        )

    split_generator = torch.Generator().manual_seed(42 + partition_id)
    local_indices = partition_indices[
        torch.randperm(len(partition_indices), generator=split_generator)
    ]
    validation_size = max(1, int(len(local_indices) * 0.2))
    train_indices = local_indices[validation_size:]
    validation_indices = local_indices[:validation_size]
    train_dataset = TensorDataset(features[train_indices], labels[train_indices])
    validation_dataset = TensorDataset(
        features[validation_indices], labels[validation_indices]
    )
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(validation_dataset, batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset(dataset_dir: str | Path | None = None):
    """Load the official UCI HAR test split for centralized evaluation."""
    dataset = _load_dataset("test", dataset_dir)
    return DataLoader(TensorDataset(*dataset.tensors[:2]), batch_size=128)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for features, labels in trainloader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(features), labels)
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
    true_positive = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)
    predicted_counts = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)
    actual_counts = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)
    with torch.no_grad():
        for features, labels in testloader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = net(features)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predictions = probabilities.max(dim=1)
            matches = predictions == labels
            loss += criterion(outputs, labels).item()
            correct += matches.sum().item()
            total_examples += labels.size(0)
            confidence_sum += confidence.sum().item()
            correct_confidence_sum += confidence[matches].sum().item()
            wrong_confidence_sum += confidence[~matches].sum().item()
            true_positive += torch.bincount(labels[matches], minlength=NUM_CLASSES)
            predicted_counts += torch.bincount(predictions, minlength=NUM_CLASSES)
            actual_counts += torch.bincount(labels, minlength=NUM_CLASSES)
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    precision_per_class = true_positive.float() / predicted_counts.clamp(min=1)
    recall_per_class = true_positive.float() / actual_counts.clamp(min=1)
    f1_per_class = (
        2 * precision_per_class * recall_per_class
        / (precision_per_class + recall_per_class).clamp(min=1e-12)
    )
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
