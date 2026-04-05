"""
Dataset processing module for GNSS interference classification.

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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json

import numpy as np
from scipy.signal import stft
from skimage.transform import resize

from process_oakbat import *
from process_swinney import *
from feature_extraction import *


# Constants
N_FEATURES:  int   = 8     # Length of the feature vector from compute_features()
SAMPLE_RATE: float = 5e6   # 5 MHz — shared sampling frequency for both datasets


@dataclass
class Segment:
    """
    A labelled IQ window with an optional pre-computed feature vector.

    Used in the non-streaming (in-memory) path. In the streaming pipeline,
    the lighter SegmentMeta is used during scanning and the IQ is read
    on demand via read_segment_iq().

    Attributes:
        data:         Complex IQ samples, shape (num_samples,), dtype complex64.
        label:        Unified class label string (e.g. 'clean', 'jam_chirp').
        source_file:  Path to the originating .bin or .mat file.
        scenario:     Scenario identifier (e.g. 'ds1', 'swinney_training').
        start_sample: Sample offset within the source file. Always 0 for
                      Swinney (whole-file segments).
        is_spoofed:   True for OAKBAT post-onset windows; False otherwise.
        features:     8-element float32 feature vector from compute_features().
                      None until explicitly computed — callers that do not
                      use features are unaffected.
    """
    data:         np.ndarray
    label:        str
    source_file:  str
    scenario:     str
    start_sample: int
    is_spoofed:   bool
    features:     Optional[np.ndarray] = field(default=None)


@dataclass
class STFTParams:
    """
    STFT configuration shared between the dataset pipeline and inference.

    All fields must match between training and deployment — changing any
    parameter produces incompatible spectrograms. The defaults produce
    128 x 128 log-magnitude images from 20 ms IQ windows at 5 MHz.

    Attributes:
        nperseg:     FFT size and analysis window length in samples.
        noverlap:    Number of overlapping samples between adjacent STFT
                     frames. Default is 75% overlap (192/256).
        window:      SciPy window function name (e.g. 'hann', 'hamming').
        output_size: (height, width) of the resized output image in pixels.
        log_scale:   If True, convert magnitude to dB: 10·log₁₀(|Z|² + ε).
        epsilon:     Floor added inside the log to avoid −∞ on silent bins.
    """
    nperseg:     int   = 256
    noverlap:    int   = 192
    window:      str   = "hann"
    output_size: tuple = (128, 128)
    log_scale:   bool  = True
    epsilon:     float = 1e-10


# Spectrogram computation
def compute_spectrogram(segment_data: np.ndarray,
                        fs: float,
                        params: STFTParams = STFTParams()) -> np.ndarray:
    """
    Compute a log-magnitude STFT spectrogram from an IQ segment.

    Applies a short-time Fourier transform, optionally converts to dB scale,
    and resizes to the target image dimensions with anti-aliasing.

    Args:
        segment_data: Complex IQ samples of shape (num_samples,).
        fs:           Sampling frequency in Hz.
        params:       STFT configuration. Must match across all pipeline calls.

    Returns:
        Float32 array of shape params.output_size (e.g. (128, 128)).
        Values are in dB if params.log_scale is True, otherwise linear
        magnitude. Not yet normalised — call SpectrogramNormalizer.transform()
        before saving or feeding to the model.
    """
    _, _, Zxx = stft(segment_data, fs=fs,
                     nperseg=params.nperseg,
                     noverlap=params.noverlap,
                     window=params.window)
    mag = np.abs(Zxx)
    if params.log_scale:
        mag = 10.0 * np.log10(mag ** 2 + params.epsilon)
    return resize(mag, params.output_size,
                  anti_aliasing=True,
                  preserve_range=True).astype(np.float32)


# Spectrogram normalizer
class SpectrogramNormalizer:
    """
    Global z-score or per-image min-max normalizer for spectrograms.

    For global mode, statistics are accumulated from the training split
    only (to avoid data leakage), then applied uniformly to all splits.
    For per-image mode, each spectrogram is independently scaled to [0, 1].

    Typical usage (global mode):
        # Pass 1: fit on training spectrograms
        norm = SpectrogramNormalizer(mode="global")
        for spec in train_specs:
            norm.update(spec)
        norm.finalize()

        # Pass 2: normalize all spectrograms
        spec_norm = norm.transform(spec)

    Save/load to JSON so the same statistics can be re-used at inference
    time on the Jetson Nano without re-computing them from the dataset.
    """

    def __init__(self, mode: str = "global"):
        """
        Args:
            mode: 'global' for dataset-wide z-score normalization (recommended
                  for cross-dataset generalization); 'per_image' for per-sample
                  min-max scaling to [0, 1].
        """
        self.mode             = mode
        self._running_sum     = 0.0
        self._running_sq_sum  = 0.0
        self._count           = 0
        self.mean: Optional[float] = None
        self.std:  Optional[float] = None

    def update(self, spectrogram: np.ndarray) -> None:
        """
        Accumulate pixel-level sum and sum-of-squares for global stats.

        Must be called on training spectrograms only (before finalize).

        Args:
            spectrogram: Float array of any shape. All pixels contribute.
        """
        self._running_sum    += spectrogram.sum()
        self._running_sq_sum += (spectrogram ** 2).sum()
        self._count          += spectrogram.size

    def finalize(self) -> None:
        """
        Compute global mean and std from accumulated pixel statistics.

        Applies a floor of 1.0 to std so that constant spectrograms
        (e.g. from silent segments) do not produce division-by-zero errors.

        Raises:
            RuntimeError: If update() has not been called at least once.
        """
        if self._count == 0:
            raise RuntimeError("No data accumulated. Call update() first.")
        self.mean = self._running_sum / self._count
        self.std  = np.sqrt(self._running_sq_sum / self._count - self.mean ** 2)
        if self.std < 1e-8:
            self.std = 1.0
        print(f"[Normalizer] Global mean={self.mean:.4f}, std={self.std:.4f}")

    def transform(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Normalize a spectrogram using the fitted or per-image strategy.

        Args:
            spectrogram: Float array to normalize.

        Returns:
            Normalized float32 array of the same shape.
        """
        if self.mode == "global":
            return ((spectrogram - self.mean) / self.std).astype(np.float32)
        else:  # per_image
            lo, hi = spectrogram.min(), spectrogram.max()
            return ((spectrogram - lo) / (hi - lo + 1e-10)).astype(np.float32)

    def save(self, path: str) -> None:
        """
        Persist normalizer statistics to a JSON file.

        Args:
            path: Destination file path (e.g. 'normalization_stats.json').
        """
        stats = {"mode": self.mode,
                 "mean": float(self.mean),
                 "std":  float(self.std)}
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SpectrogramNormalizer":
        """
        Restore a SpectrogramNormalizer from a previously saved JSON file.

        Args:
            path: Path to the JSON file produced by save().

        Returns:
            SpectrogramNormalizer with mean and std populated.
        """
        with open(path) as f:
            stats = json.load(f)
        norm      = cls(mode=stats["mode"])
        norm.mean = stats["mean"]
        norm.std  = stats["std"]
        return norm


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
                         to test (test_ratio = 1 − train_ratio − val_ratio).
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
    if meta.dataset == "oakbat":
        return read_oakbat_chunk(meta.source_file, meta.start_sample,
                                 meta.num_samples)
    else:  # swinney
        return read_swinney_file(meta.source_file)


def save_dataset_streaming(splits: dict,
                            output_dir: str,
                            fs: float,
                            stft_params: STFTParams,
                            normalizer: SpectrogramNormalizer,
                            save_format: str = "npy") -> None:
    """
    Stream all splits to disk as normalised spectrograms, one segment at a
    time.

    Spectrograms are saved under:
        <output_dir>/<split>/<label>/spec_NNNNN.<ext>

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
    metadata: dict = {
        "fs": fs,
        "stft_params": {
            "nperseg":     stft_params.nperseg,
            "noverlap":    stft_params.noverlap,
            "window":      stft_params.window,
            "output_size": list(stft_params.output_size),
            "log_scale":   stft_params.log_scale,
        },
        "splits": {},
    }

    for split_name, segments in splits.items():
        split_dir = out / split_name
        counters: dict[str, int] = {}
        metadata["splits"][split_name] = {}

        for i, meta in enumerate(segments):
            iq   = read_segment_iq(meta)
            spec = compute_spectrogram(iq, fs, stft_params)
            spec = normalizer.transform(spec)

            label_dir = split_dir / meta.label
            label_dir.mkdir(parents=True, exist_ok=True)

            idx                  = counters.get(meta.label, 0) + 1
            counters[meta.label] = idx

            if save_format == "npy":
                np.save(str(label_dir / f"spec_{idx:05d}.npy"), spec)
            elif save_format == "png":
                from PIL import Image
                img = ((spec - spec.min()) /
                       (spec.max() - spec.min() + 1e-10) * 255)
                Image.fromarray(img.astype(np.uint8)).save(
                    str(label_dir / f"spec_{idx:05d}.png"))

            if (i + 1) % 500 == 0:
                print(f"  [{split_name}] Saved {i + 1}/{len(segments)}")

        for label, count in counters.items():
            metadata["splits"][split_name][label] = count
        print(f"  [{split_name}] Done: {sum(counters.values())} spectrograms")

    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    normalizer.save(str(out / "normalization_stats.json"))
    print(f"\n[Save] Dataset written to {out}")

# Normalization helpers
def compute_normalization_streaming(segments: list,
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


def compute_joint_normalization(oakbat_train_meta: list,
                                swinney_train_meta: list,
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
    combined = oakbat_train_meta + swinney_train_meta
    print(f"[JointNorm] Computing stats across {len(combined)} segments "
          f"({len(oakbat_train_meta)} OAKBAT + {len(swinney_train_meta)} Swinney)")

    normalizer = compute_normalization_streaming(combined, fs, stft_params)

    if output_path:
        normalizer.save(output_path)
        print(f"[JointNorm] Stats saved to {output_path}")

    return normalizer