"""
Model-agnostic training and evaluation utilities for GNSS interference
classification.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import torch
import torch.nn as nn
from torch import no_grad
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    fusion: bool = False,
) -> tuple[float, float]:
    """
    Run one full training epoch over the provided dataloader. \n
    Training: iterate over batches -> compute forward pass & backpropagate -> update weights.\n
    Metrics are accumulated across all batches and returned as epoch-level averages.

    Args:
        model: The nn.Module to train. Must already be on `device`.
        dataloader: DataLoader yielding (inputs, labels) batches.
        criterion: Loss function (e.g. nn.CrossEntropyLoss()).
        optimizer: Optimiser instance (e.g. torch.optim.Adam).
        device: torch.device to move tensors to (cpu / cuda).

    Returns:
        average_loss: Mean per-sample loss across the full epoch.
        accuracy: Fraction of correctly classified samples.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        if fusion:
            (spectrograms, features), labels = batch
            spectrograms = spectrograms.to(device)
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(spectrograms, features)
        else:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # Computed after the loop so they reflect the full epoch, not the last batch.
    average_loss = running_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    fusion: bool = False,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Run an inference pass over the provided dataloader (no gradient updates).\n
    Set model to evaluation mode, collet per-sample prediction and ground-truth labels so caller can compute 
    desired performance metrics.

    Args:
        model:      The nn.Module to evaluate. Must already be on `device`.
        dataloader: DataLoader yielding (inputs, labels) batches.
        criterion:  Loss function used to compute the reported loss.
        device:     torch.device to move tensors to (cpu / cuda).

    Returns:
        average_loss: Mean per-sample loss across the full dataset.
        accuracy:     Fraction of correctly classified samples.
        all_preds:    int32 array of shape (N,) with predicted class indices.
        all_labels:   int32 array of shape (N,) with ground-truth class indices.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with no_grad():
        for batch in dataloader:
            if fusion:
                (spectrograms, features), labels = batch
                spectrograms = spectrograms.to(device)
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(spectrograms, features)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return (
        running_loss / total,
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
    )


def plot_training_curves(
    history: dict[str, list[float]], 
    output_dir: str
) -> None:
    """
    Save a two-panel loss and accuracy plot for a completed training run.

    Args:
        history:    Dict with keys 'train_loss', 'val_loss', 'train_acc',
                    'val_acc', each mapping to a list of per-epoch floats.
        output_dir: Directory where 'training_curves.png' will be written.
                    Must already exist.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epoch_range = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epoch_range, history["train_loss"], "b-", label="Train")
    ax1.plot(epoch_range, history["val_loss"],   "r-", label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epoch_range, history["train_acc"], "b-", label="Train")
    ax2.plot(epoch_range, history["val_acc"],   "r-", label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved training curves to {path}")
 
 
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_dir: str,
    title: str = "Confusion Matrix",
) -> None:
    """
    Save a labelled confusion matrix for a test-set evaluation.\n
    Figure size scales with the number of classes so labels remain readable 
    regardless of taxonomy size.

    Args:
        y_true:       Ground-truth class indices, shape (N,).
        y_pred:       Predicted class indices, shape (N,).
        class_names:  Ordered list of human-readable class names matching
                      the integer indices in y_true / y_pred.
        output_dir:   Directory where 'confusion_matrix.png' will be written.
                      Must already exist.
        title:        Title string displayed above the matrix.
    """

    n_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    fig, ax = plt.subplots(
        figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.8))
    )
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion matrix to {path}")