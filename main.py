"""
GNSS Interference Classification — Main Pipeline

Usage:
    python main.py                          # full pipeline (prepare + train)
    python main.py --skip-prepare           # train only (spectrograms exist)
    python main.py --prepare-only           # prepare only (no training)
"""

import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# Project modules
from modules.common.types import STFTParams, SAMPLE_RATE
from modules.common.spectrogram import SpectrogramNormalizer

from modules.dataset_module.process_oakbat import (
    load_oakbat_iq, segment_signal, scan_oakbat_segments, ALL_SCENARIOS,
)
from modules.dataset_module.process_swinney import load_swinney_segments
from modules.dataset_module.dataset_main import (
    create_splits, compute_joint_normalization, save_dataset_streaming,
)

from modules.nn_module.nn_main import (
    train_one_epoch, evaluate, plot_training_curves, plot_confusion_matrix,
)
from modules.nn_module.resnet18_init import build_resnet18
from modules.nn_module.mobilenetv2_init import build_mobilenetv2
from modules.nn_module.efficientnetb0_init import build_efficientnetb0
from modules.nn_module.custom_cnn_init import build_custom_cnn
from modules.nn_module.dataset import SpectrogramDataset

from modules.common.features import FeatureNormalizer
from modules.dataset_module.dataset_main import compute_feature_normalization
from modules.nn_module.fusion_model import build_fusion_model

# Paths
OAKBAT_DATASET_PATH  = "./modules/dataset_module/datasets/OakbatSpoofing"
SWINNEY_DATASET_PATH = "./modules/dataset_module/datasets/SwinneyJamming"
SPECTROGRAM_DIR      = "./Output/combined_spectrograms"
OUTPUT_DIR           = "./Output/results_11classes_regularized_fusion"


# Prepare dataset
def prepare_datasets(oakbat_dir: str = OAKBAT_DATASET_PATH,
                     swinney_dir: str = SWINNEY_DATASET_PATH,
                     output_dir: str = SPECTROGRAM_DIR,
                     max_per_class: int = 1000):
    """
    Load raw IQ from both datasets, segment, split, normalize, and save
    spectrograms to disk.
    """
    stft_params = STFTParams()

    # ── Step 1: Scan OAKBAT (metadata only, no IQ loaded) ─────────
    print("Scanning OAKBAT segments (metadata only)...")
    oakbat_segments = scan_oakbat_segments(oakbat_dir)

    # ── Step 2: Load Swinney ───────────────────────────────────────────
    swinney_segments = []
    for split in ["training", "testing"]:
        try:
            segs = load_swinney_segments(swinney_dir, split)
            swinney_segments.extend(segs)
        except FileNotFoundError:
            print(f"  [Note] Swinney {split} not found, skipping")

    print(f"\nTotal: {len(oakbat_segments)} OAKBAT + "
          f"{len(swinney_segments)} Swinney segments")

    # ── Step 3: Combine and split ──────────────────────────────────────
    combined = oakbat_segments + swinney_segments
    splits = create_splits(combined, balance_classes=True,
                           max_per_class=max_per_class)

    # ── Step 4: Joint spectrogram normalization (training split only) ──
    os.makedirs(output_dir, exist_ok=True)
    spec_normalizer = compute_joint_normalization(
        [s for s in splits["train"] if s.dataset == "oakbat"],
        [s for s in splits["train"] if s.dataset == "swinney"],
        stft_params,
        output_path=os.path.join(output_dir, "normalization_stats.json"),
    )

    # ── Step 5: Feature normalization (training split only) ────────────
    feat_normalizer = compute_feature_normalization(
        splits["train"],
        fs=SAMPLE_RATE,
        output_path=os.path.join(output_dir, "feature_norm_stats.json"),
    )

    # ── Step 6: Save spectrograms + features ───────────────────────────
    save_dataset_streaming(splits, output_dir, SAMPLE_RATE,
                           stft_params, spec_normalizer, feat_normalizer)
    

# Training and evaluation
def run_training(
    data_dir: str = SPECTROGRAM_DIR,
    output_dir: str = OUTPUT_DIR,
    model_name: str = "resnet18",
    fusion: bool = False,
    backbone_name: str = "resnet18",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    freeze_epochs: int = 5,
    weight_decay: float = 1e-3,
    num_workers: int = 2,
    device_str: str = "auto",
):
    """
    Full training pipeline with two-phase strategy:
        1. Frozen backbone for freeze_epochs (FC head only)
        2. Full fine-tuning for remaining epochs at lr/10
    """
    os.makedirs(output_dir, exist_ok=True)
    data_path = Path(data_dir)

    # ── Device ─────────────────────────────────────────────────────────
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    print(f"\nUsing device: {device}")

    # ── Datasets ───────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_dataset = SpectrogramDataset(str(data_path / "train"),
                                       load_features=fusion)
    class_to_idx  = train_dataset.class_to_idx

    val_dataset  = SpectrogramDataset(str(data_path / "val"),  class_to_idx,
                                      load_features=fusion)
    test_dataset = SpectrogramDataset(str(data_path / "test"), class_to_idx,
                                      load_features=fusion)

    class_names = [train_dataset.idx_to_class[i]
                   for i in range(train_dataset.num_classes)]
    num_classes = train_dataset.num_classes

    print(f"\nClasses ({num_classes}): {class_names}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
          f"Test: {len(test_dataset)}")

    # ── DataLoaders ────────────────────────────────────────────────────
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                         pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    # ── Model builders ─────────────────────────────────────────────
    if fusion:
        print(f"\nBuilding fusion model with {backbone_name} backbone...")
        model = build_fusion_model(backbone_name, num_classes,
                                   freeze_backbone=True).to(device)
    else:
        MODEL_BUILDERS = {
            "resnet18":       build_resnet18,
            "mobilenetv2":    build_mobilenetv2,
            "efficientnetb0": build_efficientnetb0,
            "custom_cnn":     build_custom_cnn,
        }
        print(f"\nBuilding model: {model_name}...")
        if model_name not in MODEL_BUILDERS:
            raise ValueError(f"Unknown model: {model_name}. "
                             f"Choose from: {list(MODEL_BUILDERS.keys())}")
        model = MODEL_BUILDERS[model_name](num_classes,
                                           freeze_backbone=True).to(device)
        
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Training loop ──────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_epoch   = 0

    print(f"\n{'='*60}")
    print(f"Training: {epochs} epochs ({freeze_epochs} frozen + "
          f"{epochs - freeze_epochs} fine-tune)")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        if epoch == freeze_epochs + 1:
            print(f"\n  [Epoch {epoch}] Unfreezing backbone, "
                  f"LR → {lr / 10:.1e}")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=lr / 10,
                                  weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - freeze_epochs)
        elif epoch == 1:
            if fusion:
                # Train feature MLP + classifier during frozen phase
                head_params = list(model.feat_mlp.parameters()) + \
                              list(model.classifier.parameters())
            elif hasattr(model, 'fc'):
                head_params = model.fc.parameters()
            elif hasattr(model, 'classifier'):
                head_params = model.classifier.parameters()
            else:
                head_params = model.parameters()
            optimizer = optim.Adam(head_params, lr=lr,
                                  weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=3, gamma=0.5)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, fusion=fusion)
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device, fusion=fusion)
        scheduler.step()

        elapsed = time.time() - epoch_start
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        phase = "frozen" if epoch <= freeze_epochs else "fine-tune"
        print(f"  Epoch {epoch:3d}/{epochs} [{phase:9s}] "
              f"Train: {train_acc:.4f} ({train_loss:.4f})  "
              f"Val: {val_acc:.4f} ({val_loss:.4f})  [{elapsed:.1f}s]")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "class_to_idx": class_to_idx,
                "num_classes": num_classes,
            }, os.path.join(output_dir, "best_model.pth"))

    print(f"\nBest validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")

    # ── Plots ──────────────────────────────────────────────────────────
    plot_training_curves(history, output_dir)

    # ── Test evaluation ────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    ckpt = torch.load(os.path.join(output_dir, "best_model.pth"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, fusion=fusion)

    print(f"\n{'='*60}")
    print(f"TEST RESULTS (best model from epoch {best_epoch})")
    print(f"{'='*60}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Loss:     {test_loss:.4f}")

    report = classification_report(
        test_labels, test_preds, target_names=class_names, digits=4)
    print(f"\nClassification Report:\n{report}")

    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Best Epoch: {best_epoch}\n\n")
        f.write(report)

    plot_confusion_matrix(test_labels, test_preds, class_names, output_dir)

    # ── Save full results ──────────────────────────────────────────────
    results = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "best_epoch": best_epoch,
        "best_val_accuracy": float(best_val_acc),
        "epochs": epochs, "freeze_epochs": freeze_epochs,
        "batch_size": batch_size, "learning_rate": lr,
        "num_classes": num_classes, "class_names": class_names,
        "class_to_idx": class_to_idx, "history": history,
        "device": str(device),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    model_size = os.path.getsize(
        os.path.join(output_dir, "best_model.pth")) / (1024 * 1024)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model size: {model_size:.1f} MB, "
          f"Parameters: {total_params:,}")
    print(f"  Results saved to {output_dir}/")

    return model, results


def main():
    parser = argparse.ArgumentParser(
        description="GNSS Interference Classification Pipeline")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip dataset preparation (spectrograms exist)")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only prepare dataset, do not train")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "mobilenetv2", 
                                 "efficientnetb0", "custom_cnn"],
                        help="Model architecture to train")
    parser.add_argument("--fusion", action="store_true",
                        help="Use spectrogram + feature fusion model")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "mobilenetv2", "efficientnetb0"],
                        help="CNN backbone for fusion model")
    args = parser.parse_args()

    if not args.skip_prepare:
        prepare_datasets()

    if not args.prepare_only:
        if args.fusion:
            output_dir = os.path.join(OUTPUT_DIR, f"fusion_{args.backbone}")
        else:
            output_dir = os.path.join(OUTPUT_DIR, args.model)
        run_training(model_name=args.model, fusion=args.fusion,
                     backbone_name=args.backbone, output_dir=output_dir,
                     epochs=args.epochs, batch_size=args.batch_size,
                     lr=args.lr, device_str=args.device)


if __name__ == "__main__":
    main()