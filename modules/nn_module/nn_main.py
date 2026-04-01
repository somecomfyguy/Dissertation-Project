"""
Main function for neural network module
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from resnet18_init import build_resnet18


class SpectrogramDataset(Dataset):
    """
    Loads pre-computed .npy spectrograms from a split directory.
 
    Each spectrogram is a 128x128 float32 array (single-channel).
    ResNet-18 expects 3-channel input, so we replicate across channels.
    """
 
    def __init__(self, split_dir: str, class_to_idx: dict = None):
        """
        Args:
            split_dir:     Path to train/, val/, or test/ directory
            class_to_idx:  Optional pre-defined label mapping. If None,
                           labels are inferred from subdirectory names
                           (sorted alphabetically for reproducibility).
        """
        self.samples = []  # List of (filepath, class_index)
        split_path = Path(split_dir)
 
        # Discover classes from subdirectories
        class_dirs = sorted([
            d for d in split_path.iterdir() if d.is_dir()
        ])
 
        if class_to_idx is None:
            self.class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
        else:
            self.class_to_idx = class_to_idx
 
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
 
        # Collect all .npy files
        for class_dir in class_dirs:
            label = class_dir.name
            if label not in self.class_to_idx:
                print(f"  [WARN] Skipping unknown class dir: {label}")
                continue
            idx = self.class_to_idx[label]
            for npy_file in sorted(class_dir.glob("*.npy")):
                self.samples.append((str(npy_file), idx))
 
        print(f"  Loaded {len(self.samples)} samples, "
              f"{self.num_classes} classes from {split_dir}")
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, index):
        filepath, label = self.samples[index]
        spec = np.load(filepath).astype(np.float32)
 
        # Shape: (128, 128) → (3, 128, 128) for ResNet
        # Replicate single channel across RGB
        tensor = torch.from_numpy(spec).unsqueeze(0).expand(3, -1, -1)
 
        return tensor, label
    
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss and accuracy."""
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
 
    return running_loss / total, correct / total


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


def run_training(
    data_dir: str,
    output_dir: str = "./results",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    freeze_epochs: int = 5,
    weight_decay: float = 1e-4,
    num_workers: int = 2,
    device_str: str = "auto",
):
    """
    Full training pipeline.
 
    Strategy:
        1. Train with frozen backbone for `freeze_epochs` (only FC layer)
        2. Unfreeze and fine-tune entire network for remaining epochs at lr/10
        3. Evaluate on test set with confusion matrix and classification report
 
    Args:
        data_dir:       Root of processed spectrogram dataset
        output_dir:     Where to save model, plots, and metrics
        epochs:         Total training epochs
        batch_size:     Batch size for all dataloaders
        lr:             Initial learning rate (for frozen phase)
        freeze_epochs:  Epochs to train with frozen backbone before unfreezing
        weight_decay:   L2 regularization
        num_workers:    DataLoader workers (set to 0 if you get multiprocessing errors)
        device_str:     "auto", "cuda", "mps", or "cpu"
    """
    os.makedirs(output_dir, exist_ok=True)
    data_path = Path(data_dir)
 
    # --- Device ---
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")
 
    # --- Datasets ---
    print("\nLoading datasets...")
    train_dataset = SpectrogramDataset(str(data_path / "train"))
    class_to_idx = train_dataset.class_to_idx  # Use same mapping for all splits
 
    val_dataset = SpectrogramDataset(str(data_path / "val"), class_to_idx)
    test_dataset = SpectrogramDataset(str(data_path / "test"), class_to_idx)
 
    class_names = [train_dataset.idx_to_class[i]
                   for i in range(train_dataset.num_classes)]
    num_classes = train_dataset.num_classes
 
    print(f"\nClasses ({num_classes}): {class_names}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
          f"Test: {len(test_dataset)}")
 
    # --- DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=(device.type == "cuda"))
 
    # --- Model ---
    print("\nBuilding model...")
    model = build_resnet18(num_classes, freeze_backbone=True)
    model = model.to(device)
 
    criterion = nn.CrossEntropyLoss()
 
    # --- Training ---
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_epoch = 0
 
    print(f"\n{'='*60}")
    print(f"Training: {epochs} epochs ({freeze_epochs} frozen + "
          f"{epochs - freeze_epochs} fine-tune)")
    print(f"{'='*60}")
 
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
 
        # --- Phase transition: unfreeze backbone ---
        if epoch == freeze_epochs + 1:
            print(f"\n  [Epoch {epoch}] Unfreezing backbone, reducing LR to {lr / 10:.1e}")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=lr / 10,
                                   weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - freeze_epochs)
        elif epoch == 1:
            # Frozen phase: only train FC layer
            optimizer = optim.Adam(model.fc.parameters(), lr=lr,
                                   weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
 
        # --- Train & validate ---
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device)
 
        scheduler.step()
 
        elapsed = time.time() - epoch_start
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
 
        phase = "frozen" if epoch <= freeze_epochs else "fine-tune"
        print(f"  Epoch {epoch:3d}/{epochs} [{phase:9s}] "
              f"Train: {train_acc:.4f} ({train_loss:.4f})  "
              f"Val: {val_acc:.4f} ({val_loss:.4f})  "
              f"[{elapsed:.1f}s]")
 
        # --- Save best model ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "class_to_idx": class_to_idx,
                "num_classes": num_classes,
            }, os.path.join(output_dir, "best_model.pth"))
 
    print(f"\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
 
    # --- Plot training curves ---
    print("\nGenerating plots...")
    plot_training_curves(history, output_dir)
 
    # --- Test evaluation ---
    print("\nEvaluating on test set...")
    checkpoint = torch.load(os.path.join(output_dir, "best_model.pth"),
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
 
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device)
 
    print(f"\n{'='*60}")
    print(f"TEST RESULTS (best model from epoch {best_epoch})")
    print(f"{'='*60}")
    print(f"  Accuracy: {test_acc:.4f} ({sum(test_preds == test_labels)}"
          f"/{len(test_labels)} correct)")
    print(f"  Loss:     {test_loss:.4f}")
 
    # --- Classification report ---
    report = classification_report(
        test_labels, test_preds, target_names=class_names, digits=4)
    print(f"\nClassification Report:\n{report}")
 
    # Save report to file
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Best Epoch: {best_epoch}\n\n")
        f.write(report)
 
    # --- Confusion matrix ---
    plot_confusion_matrix(test_labels, test_preds, class_names, output_dir)
 
    # --- Save full results ---
    results = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "best_epoch": best_epoch,
        "best_val_accuracy": float(best_val_acc),
        "epochs": epochs,
        "freeze_epochs": freeze_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "num_classes": num_classes,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "history": history,
        "device": str(device),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
 
    print(f"\nAll results saved to {output_dir}/")
    print(f"  best_model.pth           — model checkpoint")
    print(f"  training_curves.png      — loss & accuracy plots")
    print(f"  confusion_matrix.png     — test set confusion matrix")
    print(f"  classification_report.txt — per-class precision/recall/F1")
    print(f"  results.json             — full metrics & hyperparameters")
 
    # --- Model size (for Jetson Nano planning) ---
    model_size_mb = os.path.getsize(
        os.path.join(output_dir, "best_model.pth")) / (1024 * 1024)
    print(f"\n  Model checkpoint size: {model_size_mb:.1f} MB")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
 
    return model, results