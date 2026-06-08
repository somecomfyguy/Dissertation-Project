"""
Cross-dataset TEXBAT evaluation — zero-shot spoofing generalisation.

Applies a pre-trained model to TEXBAT segments without any retraining or
fine-tuning. The key operational difference vs main.py is that TEXBAT
uses the SAVED normaliser statistics from the training run — computing
fresh statistics on TEXBAT would leak target-domain information and
defeat the point of zero-shot evaluation.

Workflow:
    1. Scan TEXBAT into Segment metadata (no IQ loaded).
    2. Load saved SpectrogramNormalizer and FeatureNormalizer from the
       training output directory.
    3. For each segment: read + 5x-decimate IQ → STFT → apply saved
       normalisers → save spectrogram and feature vector to disk.
    4. Load the saved model checkpoint with its embedded class_to_idx.
    5. Build a SpectrogramDataset over the saved TEXBAT spectrograms
       using the training class_to_idx — so class indices match exactly.
    6. Run inference, report per-class metrics. TEXBAT only covers 6 of
       the 11 training classes; the report shows both a TEXBAT-only view
       and a full 11-class view (to surface any cross-contamination into
       jamming classes).

Usage:
    # Full pipeline (prepare + eval) for fusion_custom_cnn (current best)
    python eval_crossdataset.py --fusion --backbone custom_cnn

    # Eval only (spectrograms already saved)
    python eval_crossdataset.py --skip-prepare --fusion --backbone custom_cnn

    # Eval a non-fusion model
    python eval_crossdataset.py --skip-prepare --model resnet18

    # Dev mode: cap segments per class for quick turnaround
    python eval_crossdataset.py --max-per-class 500 --fusion --backbone custom_cnn
"""
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from modules.common.types import STFTParams, SAMPLE_RATE
from modules.common.spectrogram import compute_spectrogram, SpectrogramNormalizer
from modules.common.features import compute_features, FeatureNormalizer

from modules.dataset_module.process_texbat import scan_texbat_segments
from modules.dataset_module.dataset_main import read_segment_iq

from modules.nn_module.dataset import SpectrogramDataset
from modules.nn_module.nn_main import evaluate, plot_confusion_matrix
from modules.nn_module.resnet18_init import build_resnet18
from modules.nn_module.mobilenetv2_init import build_mobilenetv2
from modules.nn_module.efficientnetb0_init import build_efficientnetb0
from modules.nn_module.custom_cnn_init import build_custom_cnn
from modules.nn_module.fusion_model import build_fusion_model


# Paths
TEXBAT_DATASET_PATH = "./modules/dataset_module/datasets/TexbatSpoofing"
TRAINED_NORM_DIR    = "./Output/combined_spectrograms"
TEXBAT_SPEC_DIR     = "./Output/texbat_spectrograms_holdout"
TRAINED_MODEL_BASE  = "./Output/results_11classes_regularized_"
RESULTS_BASE_DIR    = "./Output/texbat_eval_results_holdout"


def _load_holdout_filter(dataset_key: str,
                         split_path: str = "./crossdataset_splits.json"):
    """
    Returns a predicate is_holdout(segment) -> bool using the split index.
    A segment is HOLDOUT if start_sample > threshold for its (scenario, label).
    Holdout segments were never eligible for training, so evaluating on them
    only is leakage-free.
    """
    with open(split_path) as f:
        index = json.load(f)
    thresholds = index[dataset_key]

    def is_holdout(seg) -> bool:
        key = f"{seg.scenario}|{seg.label}"
        thr = thresholds.get(key)
        if thr is None:
            print(f"  [WARN] No split threshold for '{key}', keeping segment")
            return True
        return seg.start_sample > thr

    return is_holdout


def prepare_texbat(texbat_dir: str = TEXBAT_DATASET_PATH,
                   trained_norm_dir: str = TRAINED_NORM_DIR,
                   output_dir: str = TEXBAT_SPEC_DIR,
                   norm_mode: str = "global",
                   max_per_class: int = None,) -> None:
    """
    Build TEXBAT test-set spectrograms + features on disk using the SAVED
    training normalisers. No train/val/test split — TEXBAT is a zero-shot
    test set, not part of model training.
    """
    stft_params = STFTParams()

    spec_norm_path = os.path.join(trained_norm_dir, "normalization_stats.json")
    feat_norm_path = os.path.join(trained_norm_dir, "feature_norm_stats.json")

    print(f"[Prep] Loading spectrogram normaliser from {spec_norm_path}")
    spec_normalizer = SpectrogramNormalizer.load(spec_norm_path)

    print(f"[Prep] Loading feature normaliser    from {feat_norm_path}")
    feat_normalizer = FeatureNormalizer.load(feat_norm_path)

    print(f"\n[Prep] Scanning TEXBAT...")
    segments = scan_texbat_segments(texbat_dir)
    is_holdout = _load_holdout_filter("texbat")
    before = len(segments)
    segments = [s for s in segments if is_holdout(s)]
    print(f"[Prep] Holdout filter: {before} -> {len(segments)} segments")

    # Optional subsampling for development
    if max_per_class is not None:
        rng = np.random.default_rng(42)
        by_label: dict[str, list] = defaultdict(list)
        for s in segments:
            by_label[s.label].append(s)

        subsampled = []
        for label, segs in by_label.items():
            idx = rng.permutation(len(segs))[:max_per_class]
            subsampled.extend(segs[i] for i in idx)
        segments = subsampled
        print(f"[Prep] Subsampled to {len(segments)} segments "
              f"(cap {max_per_class}/class)")

    output_path = Path(output_dir)
    counters: dict[str, int] = {}

    print(f"\n[Prep] Processing {len(segments)} segments...")
    for i, meta in enumerate(segments):
        iq   = read_segment_iq(meta)
        spec = compute_spectrogram(iq, SAMPLE_RATE, stft_params)
        if norm_mode == "perimage":
            lo, hi = spec.min(), spec.max()
            spec = ((spec - lo) / (hi - lo + 1e-10)).astype(np.float32)
        else:
            spec = spec_normalizer.transform(spec)
        feat = compute_features(iq, SAMPLE_RATE)
        feat = feat_normalizer.transform(feat)

        label_dir = output_path / meta.label
        label_dir.mkdir(parents=True, exist_ok=True)

        idx = counters.get(meta.label, 0) + 1
        counters[meta.label] = idx

        np.save(str(label_dir / f"spec_{idx:05d}.npy"), spec)
        np.save(str(label_dir / f"feat_{idx:05d}.npy"), feat)

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(segments)}]")

    metadata = {
        "fs":                SAMPLE_RATE,
        "source_dataset":    "texbat",
        "num_segments":      sum(counters.values()),
        "per_class":         counters,
        "normaliser_source": trained_norm_dir,
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[Prep] Done. Per-class counts: {counters}")
    print(f"       Output: {output_path}")


def _build_model(args, num_classes: int) -> nn.Module:
    """Dispatch to the correct model builder based on CLI args."""
    if args.fusion:
        return build_fusion_model(backbone_name=args.backbone,
                                  num_classes=num_classes,
                                  freeze_backbone=False)
    builders = {
        "resnet18":       build_resnet18,
        "mobilenetv2":    build_mobilenetv2,
        "efficientnetb0": build_efficientnetb0,
        "custom_cnn":     build_custom_cnn,
    }
    return builders[args.model](num_classes=num_classes,
                                freeze_backbone=False)


def run_texbat_inference(args, spec_dir: str) -> dict:
    """
    Load a trained checkpoint, run inference over TEXBAT spectrograms,
    produce classification report + confusion matrix + predictions.npz.
    """
    subdir = f"fusion_{args.backbone}" if args.fusion else args.model

    checkpoint_path = os.path.join(args.trained_base, subdir, "best_model.pth")
    suffix = "_perimage" if args.norm_mode == "perimage" else ""
    mix_tag = os.path.basename(args.trained_base.rstrip("/"))
    output_dir = os.path.join(RESULTS_BASE_DIR + suffix + "_" + mix_tag, subdir)
    os.makedirs(output_dir, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\nDevice: {device}")

    # Load checkpoint — has class_to_idx embedded
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt         = torch.load(checkpoint_path, map_location=device,
                              weights_only=False)
    class_to_idx = ckpt["class_to_idx"]
    num_classes  = ckpt["num_classes"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names  = [idx_to_class[i] for i in range(num_classes)]
    print(f"Model has {num_classes} classes")

    model = _build_model(args, num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Build TEXBAT dataset using the TRAINING class_to_idx so jamming
    # indices (0..10) map consistently even though only 6 classes have
    # directories on disk.
    print(f"\nLoading TEXBAT spectrograms from {spec_dir}")
    test_dataset = SpectrogramDataset(spec_dir,
                                      class_to_idx=class_to_idx,
                                      load_features=args.fusion)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=(device.type == "cuda"))

    # Inference
    print(f"\nRunning inference over {len(test_dataset)} segments...")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    test_loss, test_acc, preds, labels = evaluate(
        model, test_loader, criterion, device, fusion=args.fusion)

    print(f"\n{'='*60}")
    print(f"TEXBAT ZERO-SHOT RESULTS — {subdir}")
    print(f"{'='*60}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Loss:     {test_loss:.4f}")
    # Majority-class baseline — a result must beat this to mean anything
    unique, counts = np.unique(labels, return_counts=True)
    majority_frac = counts.max() / counts.sum()
    majority_class = idx_to_class[unique[counts.argmax()]]
    print(f"  Majority baseline: {majority_frac:.4f} "
          f"(always predict '{majority_class}')")
    print(f"  Beats baseline:    {'YES' if test_acc > majority_frac else 'NO'}")

    # TEXBAT-only report (6 classes actually present)
    present_ids   = sorted(set(labels))
    present_names = [idx_to_class[i] for i in present_ids]
    report_texbat = classification_report(
        labels, preds,
        labels=present_ids, target_names=present_names,
        digits=4, zero_division=0)
    print(f"\n--- TEXBAT classes only ---\n{report_texbat}")

    # Full 11-class report: shows if TEXBAT segments get misrouted into
    # jamming classes (should be rare if the model generalises cleanly).
    report_full = classification_report(
        labels, preds,
        labels=list(range(num_classes)), target_names=class_names,
        digits=4, zero_division=0)
    print(f"\n--- Full 11-class view "
          f"(jamming classes have zero TEXBAT support) ---\n{report_full}")

    # Persist everything
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(f"TEXBAT Zero-Shot Evaluation — {subdir}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Samples:  {len(test_dataset)}\n")
        f.write(f"Accuracy: {test_acc:.4f}\n")
        f.write(f"Loss:     {test_loss:.4f}\n\n")
        f.write("=== TEXBAT classes only ===\n")
        f.write(report_texbat)
        f.write("\n\n=== Full 11-class ===\n")
        f.write(report_full)

    plot_confusion_matrix(labels, preds, class_names, output_dir,
                          title=f"TEXBAT Confusion — {subdir}")

    np.savez(os.path.join(output_dir, "predictions.npz"),
             labels=labels, preds=preds,
             class_names=np.array(class_names))

    results = {
        "model":                  subdir,
        "test_accuracy":          float(test_acc),
        "test_loss":              float(test_loss),
        "num_samples":            len(test_dataset),
        "texbat_classes_present": present_names,
        "majority_baseline":      float(majority_frac),
        "majority_class":         majority_class,
        "beats_baseline":         bool(test_acc > majority_frac),
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {output_dir}/")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="TEXBAT cross-dataset zero-shot evaluation")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip spectrogram generation (already on disk)")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only prepare, skip inference")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "mobilenetv2",
                                 "efficientnetb0", "custom_cnn"])
    parser.add_argument("--fusion", action="store_true")
    parser.add_argument("--backbone", type=str, default="custom_cnn",
                        choices=["resnet18", "mobilenetv2",
                                 "efficientnetb0", "custom_cnn"])
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device",      type=str, default="auto")
    parser.add_argument("--norm-mode", type=str, default="global",
                        choices=["global", "perimage"],
                        help="Spectrogram normalisation: global (saved stats) "
                        "or perimage (min-max per spectrogram)")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Subsample to this many segments per class "
                             "during prepare (for dev/testing)")
    parser.add_argument("--trained-base", type=str, default=TRAINED_MODEL_BASE,
                        help="Directory holding trained checkpoints "
                             "(e.g. ./Output/results_mix10)")
    parser.add_argument("--norm-dir", type=str, default=TRAINED_NORM_DIR,
                        help="Directory holding saved normaliser stats "
                             "(e.g. ./Output/combined_spectrograms_mix10)")
    args = parser.parse_args()

    spec_dir = TEXBAT_SPEC_DIR + ("_perimage" if args.norm_mode == "perimage" else "")
    if not args.skip_prepare:
        prepare_texbat(trained_norm_dir=args.norm_dir,
                       output_dir=spec_dir,
                       norm_mode=args.norm_mode,
                       max_per_class=args.max_per_class)

    if not args.prepare_only:
        run_texbat_inference(args, spec_dir)


if __name__ == "__main__":
    main()