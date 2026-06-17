"""
Cross-dataset GATEMAN evaluation — zero-shot jamming generalisation
with JSR sweep.

Tests whether jamming patterns learned from Swinney (synthetic) generalise
to GATEMAN (lab-recorded). The key addition over the TEXBAT eval is the
JSR sweep: we prepare and evaluate at multiple jammer-to-signal ratios
to produce accuracy-vs-JSR curves per jamming class.

Usage:
    # Full pipeline at default JSR (20 dB)
    python eval_crossdataset_gateman.py --fusion --backbone custom_cnn

    # JSR sweep (prepare + eval at 10, 20, 30 dB)
    python eval_crossdataset_gateman.py --jsr 10 20 30 --fusion --backbone custom_cnn

    # Eval only (spectrograms exist)
    python eval_crossdataset_gateman.py --skip-prepare --jsr 20 --model custom_cnn
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

from modules.dataset_module.process_gateman import scan_gateman_segments
from modules.dataset_module.dataset_main import read_segment_iq

from modules.nn_module.dataset import SpectrogramDataset
from modules.nn_module.nn_main import evaluate, plot_confusion_matrix
from modules.nn_module.resnet18_init import build_resnet18
from modules.nn_module.mobilenetv2_init import build_mobilenetv2
from modules.nn_module.efficientnetb0_init import build_efficientnetb0
from modules.nn_module.custom_cnn_init import build_custom_cnn
from modules.nn_module.fusion_model import build_fusion_model


# Paths
GATEMAN_DATASET_PATH = "./modules/dataset_module/datasets/GatemanJamming"
TRAINED_NORM_DIR     = "./Output/combined_spectrograms"
GATEMAN_SPEC_BASE    = "./Output/gateman_spectrograms_holdout"
TRAINED_MODEL_BASE   = "./Output/results_11classes_regularized_"
RESULTS_BASE_DIR     = "./Output/gateman_eval_results_holdout"


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


def prepare_gateman(jsr_db: float,
                    gateman_dir: str = GATEMAN_DATASET_PATH,
                    trained_norm_dir: str = TRAINED_NORM_DIR,
                    output_base: str = GATEMAN_SPEC_BASE,
                    norm_mode: str = "global",
                    max_per_class: int = None,) -> str:
    """
    Prepare GATEMAN spectrograms at a specific JSR using saved normalisers.
    Returns the output directory path.
    """
    output_dir = os.path.join(output_base, f"jsr_{int(jsr_db)}dB")
    stft_params = STFTParams()

    spec_normalizer = SpectrogramNormalizer.load(
        os.path.join(trained_norm_dir, "normalization_stats.json"))
    feat_norm = FeatureNormalizer.load(
        os.path.join(trained_norm_dir, "feature_norm_stats.json"))

    print(f"\n[Prep] Scanning GATEMAN at JSR={jsr_db} dB...")
    segments = scan_gateman_segments(gateman_dir, jsr_db=jsr_db)
    is_holdout = _load_holdout_filter("gateman")
    before = len(segments)
    segments = [s for s in segments if is_holdout(s)]
    print(f"[Prep] Holdout filter: {before} -> {len(segments)} segments")

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

    print(f"[Prep] Processing {len(segments)} segments...")
    for i, meta in enumerate(segments):
        iq   = read_segment_iq(meta)
        spec = compute_spectrogram(iq, SAMPLE_RATE, stft_params)
        spec = spec_normalizer.transform(spec)
        if norm_mode == "perimage":
            lo, hi = spec.min(), spec.max()
            spec = ((spec - lo) / (hi - lo + 1e-10)).astype(np.float32)
        else:
            spec = spec_normalizer.transform(spec)
        feat = compute_features(iq, SAMPLE_RATE)
        feat = feat_norm.transform(feat)

        label_dir = output_path / meta.label
        label_dir.mkdir(parents=True, exist_ok=True)

        idx = counters.get(meta.label, 0) + 1
        counters[meta.label] = idx

        np.save(str(label_dir / f"spec_{idx:05d}.npy"), spec)
        np.save(str(label_dir / f"feat_{idx:05d}.npy"), feat)

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(segments)}]")

    metadata = {
        "fs": SAMPLE_RATE,
        "source_dataset": "gateman",
        "jsr_db": jsr_db,
        "num_segments": sum(counters.values()),
        "per_class": counters,
        "normaliser_source": trained_norm_dir,
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[Prep] Done (JSR={jsr_db} dB). Per-class: {counters}")
    return output_dir


def _build_model(args, num_classes):
    if args.fusion:
        return build_fusion_model(backbone_name=args.backbone,
                                  num_classes=num_classes,
                                  freeze_backbone=False)
    builders = {
        "resnet18": build_resnet18, "mobilenetv2": build_mobilenetv2,
        "efficientnetb0": build_efficientnetb0, "custom_cnn": build_custom_cnn,
    }
    return builders[args.model](num_classes=num_classes, freeze_backbone=False)


def run_gateman_inference(args, jsr_db: float, spec_dir: str) -> dict:
    """Run inference on GATEMAN spectrograms at a specific JSR."""
    subdir = f"fusion_{args.backbone}" if args.fusion else args.model
    suffix = "_perimage" if args.norm_mode == "perimage" else ""
    mix_tag = os.path.basename(args.trained_base.rstrip("/"))
    output_dir = os.path.join(RESULTS_BASE_DIR + suffix + "_" + mix_tag, subdir,
                              f"jsr_{int(jsr_db)}dB")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu") if args.device == "auto" \
             else torch.device(args.device)

    checkpoint_path = os.path.join(args.trained_base, subdir, "best_model.pth")
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_to_idx = ckpt["class_to_idx"]
    num_classes  = ckpt["num_classes"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names  = [idx_to_class[i] for i in range(num_classes)]

    model = _build_model(args, num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_dataset = SpectrogramDataset(spec_dir, class_to_idx=class_to_idx,
                                      load_features=args.fusion)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))

    print(f"\nInference: {len(test_dataset)} segments at JSR={jsr_db} dB...")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    test_loss, test_acc, preds, labels = evaluate(
        model, test_loader, criterion, device, fusion=args.fusion)

    present_ids   = sorted(set(labels))
    present_names = [idx_to_class[i] for i in present_ids]

    print(f"\n{'='*60}")
    print(f"GATEMAN RESULTS — {subdir} — JSR={jsr_db} dB")
    print(f"{'='*60}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Loss:     {test_loss:.4f}")
    unique, counts = np.unique(labels, return_counts=True)
    majority_frac = counts.max() / counts.sum()
    majority_class = idx_to_class[unique[counts.argmax()]]
    print(f"  Majority baseline: {majority_frac:.4f} "
          f"(always predict '{majority_class}')")
    print(f"  Beats baseline:    {'YES' if test_acc > majority_frac else 'NO'}")

    report = classification_report(
        labels, preds, labels=present_ids,
        target_names=present_names, digits=4, zero_division=0)
    print(report)

    report_full = classification_report(
        labels, preds, labels=list(range(num_classes)),
        target_names=class_names, digits=4, zero_division=0)

    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(f"GATEMAN — {subdir} — JSR={jsr_db} dB\n{'='*60}\n")
        f.write(f"Samples:  {len(test_dataset)}\n")
        f.write(f"Accuracy: {test_acc:.4f}\nLoss: {test_loss:.4f}\n\n")
        f.write(f"=== GATEMAN classes ===\n{report}\n")
        f.write(f"=== Full 11-class ===\n{report_full}\n")

    n_classes = len(class_names)
    plot_confusion_matrix(labels, preds, class_names, output_dir,
                          title=f"GATEMAN — {subdir} — JSR={jsr_db} dB")

    np.savez(os.path.join(output_dir, "predictions.npz"),
             labels=labels, preds=preds,
             class_names=np.array(class_names))

    results = {
        "model": subdir, "jsr_db": jsr_db,
        "test_accuracy": float(test_acc), "test_loss": float(test_loss),
        "num_samples": len(test_dataset),
        "classes_present": present_names,
        "majority_baseline": float(majority_frac),
        "majority_class":    majority_class,
        "beats_baseline":    bool(test_acc > majority_frac),
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved to {output_dir}/")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="GATEMAN cross-dataset jamming evaluation with JSR sweep")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--jsr", type=float, nargs="+", default=[20.0],
                        help="JSR values in dB (e.g. --jsr 10 20 30)")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "mobilenetv2",
                                 "efficientnetb0", "custom_cnn"])
    parser.add_argument("--fusion", action="store_true")
    parser.add_argument("--backbone", type=str, default="custom_cnn",
                        choices=["resnet18", "mobilenetv2",
                                 "efficientnetb0", "custom_cnn"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--norm-mode", type=str, default="global",
                        choices=["global", "perimage"])
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--trained-base", type=str, default=TRAINED_MODEL_BASE,
                        help="Directory holding trained checkpoints")
    parser.add_argument("--norm-dir", type=str, default=TRAINED_NORM_DIR,
                        help="Directory holding saved normaliser stats")
    args = parser.parse_args()

    spec_base = GATEMAN_SPEC_BASE + ("_perimage" 
                                     if args.norm_mode == "perimage" else "")

    all_results = []

    for jsr in args.jsr:
        if not args.skip_prepare:
            spec_dir = prepare_gateman(jsr_db=jsr,
                                       trained_norm_dir=args.norm_dir,
                                       output_base=spec_base,
                                       norm_mode=args.norm_mode,
                                       max_per_class=args.max_per_class)
        else:
            spec_dir = os.path.join(GATEMAN_SPEC_BASE, f"jsr_{int(jsr)}dB")

        if not args.prepare_only:
            result = run_gateman_inference(args, jsr, spec_dir)
            all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("JSR SWEEP SUMMARY")
        print(f"{'='*60}")
        for r in all_results:
            print(f"  JSR={r['jsr_db']:5.1f} dB  →  Accuracy={r['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()