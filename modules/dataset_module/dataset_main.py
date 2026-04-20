"""
Dataset processing orchestration for GNSS interference classification.

Handles both OAKBAT (spoofing) and Swinney (jamming) datasets in a unified
two-pass streaming pipeline optimised for the 16 GB RAM / no-dedicated-GPU
development environment:

    Pass 1 — Scan:  Walk dataset directories and collect lightweight
                    SegmentMeta objects (path + offset + label only).
                    No IQ data is loaded into memory.

    Pass 2 — Save:  Stream segments one at a time, compute the STFT
                    spectrogram, apply the shared normalizer, and write
                    to disk as .npy files. Peak memory ≈ one IQ window
                    + one spectrogram at a time.

Normalization must be fitted on the training split only (to avoid data
leakage) before pass 2. For combined training, compute_joint_normalization()
derives a single set of statistics across both datasets so that their
spectrograms occupy the same numerical range.

Output directory layout produced by save_dataset_streaming():
    <output_dir>/
        train/
            clean/           spec_00001.npy …
            jam_am/          spec_00001.npy …
            …
        val/
            …
        test/
            …
        normalization_stats.json
        metadata.json

Class taxonomy (12 classes total):
    Jamming  (Swinney): clean, jam_am, jam_chirp, jam_fm, jam_dme,
                        jam_narrowband
    Spoofing (OAKBAT):  spoof_overpowered_instant, spoof_overpowered_gradual,
                        spoof_matched_time, spoof_matched_dynamic,
                        spoof_position_push, spoof_dynamic_position
"""
from pathlib import Path
from typing import Optional
import json

import numpy as np

from modules.common.types import Segment, STFTParams, SAMPLE_RATE
from modules.common.spectrogram import compute_spectrogram, SpectrogramNormalizer
from modules.common.features import compute_features, FeatureNormalizer
from modules.dataset_module.process_oakbat import read_oakbat_chunk
from modules.dataset_module.process_swinney import read_swinney_file


# Read IQ segment
def read_segment_iq(meta: Segment) -> np.ndarray:
    """
    Load the IQ data for a segment from disk, dispatching to the correct
    reader based on the originating dataset.

    Args:
        meta: A Segment (or SegmentMeta) object with at least source_file
              and dataset fields populated.

    Returns:
        Complex64 numpy array of shape (num_samples,) for OAKBAT, or the
        full-file IQ array for Swinney (whole-file segments).
    """
    # If IQ data is already loaded in memory, use it directly
    if meta.data is not None:
        return meta.data

    if meta.dataset == "oakbat":
        return read_oakbat_chunk(meta.source_file, meta.start_sample,
                                meta.num_samples)
    elif meta.dataset == "swinney":
        return read_swinney_file(meta.source_file)
    else:
        raise ValueError(f"Unknown dataset: {meta.dataset}")
 

# Splitting dataset
def create_splits(segments: list[Segment],
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  balance_classes: bool = True,
                  max_per_class: Optional[int] = None,
                  seed: int = 42) -> dict[str, list[Segment]]:
    """
    Create stratified train / val / test splits with optional class balancing.

    Stratification is per-class: each class is split independently before
    pooling, so the class distribution is (approximately) preserved across
    all three splits.

    If balance_classes is True, all classes are down-sampled to the size of
    the smallest class (further capped by max_per_class if provided). This
    prevents the clean class — which is abundant in OAKBAT — from dominating
    the training signal.

    Args:
        segments:        Flat list of Segment objects covering all classes.
        train_ratio:     Fraction of each class assigned to the training split.
        val_ratio:       Fraction assigned to validation. The remainder goes
                         to test (test_ratio = 1 - train_ratio - val_ratio).
        balance_classes: If True, down-sample all classes to the size of the
                         smallest class before splitting.
        max_per_class:   Hard cap on samples per class (applied after balancing).
                         Useful for quick development runs.
        seed:            Random seed for reproducibility.

    Returns:
        Dict with keys 'train', 'val', 'test', each mapping to a list of
        Segment objects. Class distribution is printed for each split.
    """
    rng = np.random.default_rng(seed)

    by_label: dict[str, list[Segment]] = {}
    for seg in segments:
        by_label.setdefault(seg.label, []).append(seg)

    print("\n[Split] Class distribution before balancing:")
    for label, segs in sorted(by_label.items()):
        print(f"  {label}: {len(segs)} segments")

    if balance_classes:
        min_count = min(len(v) for v in by_label.values())
        if max_per_class is not None:
            min_count = min(min_count, max_per_class)
        for label in by_label:
            idx = rng.permutation(len(by_label[label]))[:min_count]
            by_label[label] = [by_label[label][i] for i in idx]
        print(f"[Split] Balanced to {min_count} samples per class")

    splits: dict[str, list[Segment]] = {"train": [], "val": [], "test": []}
    for label, segs in by_label.items():
        perm    = rng.permutation(len(segs))
        n       = len(segs)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        for i, idx in enumerate(perm):
            if i < n_train:
                splits["train"].append(segs[idx])
            elif i < n_train + n_val:
                splits["val"].append(segs[idx])
            else:
                splits["test"].append(segs[idx])

    for split_name, split_segs in splits.items():
        labels = [s.label for s in split_segs]
        unique, counts = np.unique(labels, return_counts=True)
        dist = ", ".join(f"{l}: {c}" for l, c in zip(unique, counts))
        print(f"[Split] {split_name}: {len(split_segs)} total ({dist})")

    return splits


# Streaming save
def save_dataset_streaming(splits: dict,
                           output_dir: str,
                           fs: float,
                           stft_params: STFTParams,
                           spec_normalizer: SpectrogramNormalizer,
                           feat_normalizer: Optional[FeatureNormalizer] = None,
                           save_format: str = "npy") -> None:
    """
    Stream all splits to disk as normalised spectrograms and (optionally)
    feature vectors, one segment at a time.

    Output layout:
        <output_dir>/<split>/<label>/spec_NNNNN.npy
        <output_dir>/<split>/<label>/feat_NNNNN.npy   (if feat_normalizer given)

    A metadata.json summary and the normalizer statistics are also written
    to output_dir.

    Args:
        splits:      Dict returned by create_splits(), keyed by split name.
        output_dir:  Root directory for the output dataset. Created if absent.
        fs:          Sampling frequency passed to compute_spectrogram().
        stft_params: STFT configuration — must match inference-time settings.
        normalizer:  Fitted SpectrogramNormalizer applied to every spectrogram.
        save_format: 'npy' (float32 array, recommended) or 'png' (uint8 image).
    """
    out = Path(output_dir)
    save_features = feat_normalizer is not None
    metadata: dict = {
        "fs": fs,
        "stft_params": {
            "nperseg":     stft_params.nperseg,
            "noverlap":    stft_params.noverlap,
            "window":      stft_params.window,
            "output_size": list(stft_params.output_size),
            "log_scale":   stft_params.log_scale,
        },
        "features_saved": save_features,
        "splits": {},
    }

    for split_name, segments in splits.items():
        split_dir = out / split_name
        counters: dict[str, int] = {}
        metadata["splits"][split_name] = {}

        for i, meta in enumerate(segments):
            iq   = read_segment_iq(meta)
            spec = compute_spectrogram(iq, fs, stft_params)
            spec = spec_normalizer.transform(spec)

            label_dir = split_dir / meta.label
            label_dir.mkdir(parents=True, exist_ok=True)

            idx                  = counters.get(meta.label, 0) + 1
            counters[meta.label] = idx

            # Save spectrogram
            if save_format == "npy":
                np.save(str(label_dir / f"spec_{idx:05d}.npy"), spec)
            elif save_format == "png":
                from PIL import Image
                img = ((spec - spec.min()) /
                       (spec.max() - spec.min() + 1e-10) * 255)
                Image.fromarray(img.astype(np.uint8)).save(
                    str(label_dir / f"spec_{idx:05d}.png"))

            # Save feature vector (parallel file, same index)
            if save_features:
                feat = compute_features(iq, fs)
                feat = feat_normalizer.transform(feat)
                np.save(str(label_dir / f"feat_{idx:05d}.npy"), feat)

            if (i + 1) % 500 == 0:
                print(f"  [{split_name}] Saved {i + 1}/{len(segments)}")

        for label, count in counters.items():
            metadata["splits"][split_name][label] = count
        print(f"  [{split_name}] Done: {sum(counters.values())} spectrograms"
              + (" + features" if save_features else ""))

    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    spec_normalizer.save(str(out / "normalization_stats.json"))
    if save_features:
        feat_normalizer.save(str(out / "feature_norm_stats.json"))
    print(f"\n[Save] Dataset written to {out}")


# Normalization helpers
def compute_normalization_streaming(segments: list[Segment],
                                    fs: float,
                                    stft_params: STFTParams,
                                    ) -> SpectrogramNormalizer:
    """
    Pass 1: stream through segments to accumulate global spectrogram statistics.

    Reads one IQ window at a time, computes its spectrogram, and feeds it
    to the normalizer's online accumulator. The IQ array is immediately
    discarded, keeping peak memory to a single window.

    Should be called on the training split only. Passing val or test
    segments here would constitute data leakage.

    Args:
        segments:    List of SegmentMeta (streaming path) or Segment objects.
        fs:          Sampling frequency passed to compute_spectrogram().
        stft_params: STFT configuration.

    Returns:
        Fitted SpectrogramNormalizer (finalize() already called).
    """
    normalizer = SpectrogramNormalizer(mode="global")
    total      = len(segments)

    print(f"\n[Norm] Computing global stats from {total} training segments...")
    for i, meta in enumerate(segments):
        iq   = read_segment_iq(meta)
        spec = compute_spectrogram(iq, fs, stft_params)
        normalizer.update(spec)
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{total}")

    normalizer.finalize()
    return normalizer


def compute_joint_normalization(oakbat_train: list[Segment],
                                swinney_train: list[Segment],
                                stft_params: STFTParams,
                                fs: float = SAMPLE_RATE,
                                output_path: Optional[str] = None,
                                ) -> SpectrogramNormalizer:
    """
    Compute unified normalization statistics across both datasets' training
    segments in a single streaming pass.

    Running the two dataset pipelines independently would produce separate
    normalization stats, making their spectrograms numerically incompatible
    at training time. This function merges both training lists and derives
    a single mean and std that covers the combined dynamic range.

    Args:
        oakbat_train_meta:  Training SegmentMeta list from scan_oakbat_segments().
        swinney_train_meta: Training SegmentMeta list from scan_swinney_segments().
        stft_params:        Shared STFT configuration.
        fs:                 Sampling frequency (same for both datasets at 5 MHz).
        output_path:        If provided, save the fitted normalizer to this path
                            as a JSON file (e.g. 'combined_spectrograms/
                            normalization_stats.json').

    Returns:
        Fitted SpectrogramNormalizer covering both datasets.
    """
    combined = oakbat_train + swinney_train
    print(f"[JointNorm] Computing stats across {len(combined)} segments "
          f"({len(oakbat_train)} OAKBAT + {len(swinney_train)} Swinney)")

    normalizer = compute_normalization_streaming(combined, fs, stft_params)

    if output_path:
        normalizer.save(output_path)
        print(f"[JointNorm] Stats saved to {output_path}")

    return normalizer


def compute_feature_normalization(segments: list[Segment],
                                  fs: float = SAMPLE_RATE,
                                  output_path: Optional[str] = None,
                                  ) -> FeatureNormalizer:
    """
    Stream through training segments computing per-feature z-score stats.
    Must be called on the training split only to avoid data leakage.

    Args:
        segments:    Training split Segment list.
        fs:          Sampling frequency.
        output_path: If provided, save the fitted normalizer to this path.

    Returns:
        Fitted FeatureNormalizer.
    """
    normalizer = FeatureNormalizer()
    total      = len(segments)

    print(f"\n[FeatNorm] Computing feature stats from {total} "
          f"training segments...")
    for i, meta in enumerate(segments):
        iq   = read_segment_iq(meta)
        feat = compute_features(iq, fs)
        normalizer.update(feat)
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{total}")

    normalizer.finalize()

    if output_path:
        normalizer.save(output_path)

    return normalizer