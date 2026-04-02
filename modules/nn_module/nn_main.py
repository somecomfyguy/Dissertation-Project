"""
Main function for neural network module
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from resnet18_init import build_resnet18


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch. 

    Returns:
        average_loss: ratio of the running loss and total predictions
        accuracy: ratio of correct and total predictions 
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
 
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
 
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
 
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        average_loss = running_loss / total
        accuracy = correct / total
 
    return average_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate on a dataset. Returns average loss, accuracy, all preds and labels."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
 
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
 
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
 
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
 
    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def plot_training_curves(history: dict, output_dir: str):
    """Plot loss and accuracy curves for train/val."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
 
    epochs = range(1, len(history["train_loss"]) + 1)
 
    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Train")
    ax1.plot(epochs, history["val_loss"], "r-", label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
 
    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-", label="Train")
    ax2.plot(epochs, history["val_acc"], "r-", label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
 
    fig.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.close()
    print(f"  Saved training curves to {path}")
 
 
def plot_confusion_matrix(y_true, y_pred, class_names, output_dir: str,
                          title: str = "Confusion Matrix"):
    """Plot and save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.8)))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.close()
    print(f"  Saved confusion matrix to {path}")