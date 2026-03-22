"""
OAKBAT Processing Pipeline
===========================
Converts raw OAKBAT binary IQ files into 128x128 STFT spectrograms
organized by class, ready for CNN training.

OAKBAT dataset reference:
    Albright, A., Powers, S., Bonior, J., and Combs, F.,
    "A Tool for Furthering GNSS Security Research: The Oak Ridge
    Spoofing and Interference Test Battery (OAKBAT),"
    Proceedings of ION GNSS+ 2020.
    GPS DOI:     10.13139/ORNLNCCS/1664429
    Galileo DOI: 10.13139/ORNLNCCS/1665888
    GitHub:      https://github.com/oakbat/

Usage:
    python oakbat_pipeline.py --oakbat-dir /path/to/oakbat --output-dir ./spectrograms

Pipeline steps:
    1. Load raw binary IQ (16-bit signed int, interleaved I/Q)
    2. Segment into fixed-length windows using spoofing onset metadata
    3. Compute STFT spectrograms (log-magnitude)
    4. Resize to 128x128, normalize
    5. Save as class-labeled images/arrays with train/val/test splits
"""

import numpy as np
import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from scipy.signal import stft
from skimage.transform import resize


# ---------------------------------------------------------------------------
# OAKBAT configuration & metadata
# ---------------------------------------------------------------------------
OAKBAT_FS = 5e6  # 5 MHz complex sampling rate

@dataclass
class OakbatScenario:
    """Metadata for a single OAKBAT scenario file."""
    filename: str
    constellation: str        # "gps" or "galileo"
    scenario: str             # e.g., "ds1", "ds2", ..., "ds6"
    spoofing_class: str       # Human-readable class label
    onset_seconds: float      # Spoofing onset time in seconds from file start
    description: str = ""


# OAKBAT mirrors TEXBAT ds1-ds6 for both GPS and Galileo.
# Onset times below are APPROXIMATE — update these from the actual OAKBAT
# metadata files on GitHub (https://github.com/oakbat/) before running.
#
# The scenario descriptions match TEXBAT:
#   ds1: Instantaneous overpowered spoofing (time push)
#   ds2: Overpowered spoofing (+10 dB, time push)
#   ds3: Power-matched spoofing (+1.3 dB, time push)
#   ds4: Power-matched spoofing (dynamic)
#   ds5: Position push
#   ds6: Dynamic position push

SCENARIOS_GPS = [
    OakbatScenario("oakbat_gps_ds1.bin", "gps", "ds1",
                   "spoof_overpowered_instant", onset_seconds=120.0,
                   description="Instantaneous overpowered time push"),
    OakbatScenario("oakbat_gps_ds2.bin", "gps", "ds2",
                   "spoof_overpowered_gradual", onset_seconds=120.0,
                   description="Overpowered +10dB time push"),
    OakbatScenario("oakbat_gps_ds3.bin", "gps", "ds3",
                   "spoof_matched_time", onset_seconds=120.0,
                   description="Power-matched +1.3dB time push"),
    OakbatScenario("oakbat_gps_ds4.bin", "gps", "ds4",
                   "spoof_matched_dynamic", onset_seconds=120.0,
                   description="Power-matched dynamic"),
    OakbatScenario("oakbat_gps_ds5.bin", "gps", "ds5",
                   "spoof_position_push", onset_seconds=120.0,
                   description="Position push"),
    OakbatScenario("oakbat_gps_ds6.bin", "gps", "ds6",
                   "spoof_dynamic_position", onset_seconds=120.0,
                   description="Dynamic position push"),
]

SCENARIOS_GALILEO = [
    OakbatScenario("oakbat_gal_ds1.bin", "galileo", "ds1",
                   "spoof_overpowered_instant", onset_seconds=120.0,
                   description="Instantaneous overpowered time push"),
    OakbatScenario("oakbat_gal_ds2.bin", "galileo", "ds2",
                   "spoof_overpowered_gradual", onset_seconds=120.0,
                   description="Overpowered +10dB time push"),
    OakbatScenario("oakbat_gal_ds3.bin", "galileo", "ds3",
                   "spoof_matched_time", onset_seconds=120.0,
                   description="Power-matched +1.3dB time push"),
    OakbatScenario("oakbat_gal_ds4.bin", "galileo", "ds4",
                   "spoof_matched_dynamic", onset_seconds=120.0,
                   description="Power-matched dynamic"),
    OakbatScenario("oakbat_gal_ds5.bin", "galileo", "ds5",
                   "spoof_position_push", onset_seconds=120.0,
                   description="Position push"),
    OakbatScenario("oakbat_gal_ds6.bin", "galileo", "ds6",
                   "spoof_dynamic_position", onset_seconds=120.0,
                   description="Dynamic position push"),
]

ALL_SCENARIOS = SCENARIOS_GPS + SCENARIOS_GALILEO


# ---------------------------------------------------------------------------
# Step 1: Load raw IQ from OAKBAT binary files
# ---------------------------------------------------------------------------
def load_oakbat_iq(filepath: str, fs: float = OAKBAT_FS,
                   max_duration_s: Optional[float] = None) -> np.ndarray:
    """
    Load OAKBAT raw binary IQ file.

    Format: interleaved 16-bit signed integers (I, Q, I, Q, ...)
    Returns: complex64 numpy array

    Args:
        filepath:        Path to .bin file
        fs:              Sampling frequency (Hz)
        max_duration_s:  If set, only load this many seconds from the start
    """
    max_samples = None
    if max_duration_s is not None:
        max_samples = int(max_duration_s * fs)

    raw = np.fromfile(filepath, dtype=np.int16)

    if max_samples is not None:
        raw = raw[:max_samples * 2]  # *2 because interleaved I/Q

    # Reshape to (N, 2) and combine into complex
    raw = raw[:len(raw) - len(raw) % 2]  # Ensure even length
    iq = raw.reshape(-1, 2)
    signal = iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)

    return signal


# ---------------------------------------------------------------------------
# Step 2: Segment into fixed-length windows with class labels
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    """A labeled IQ segment."""
    data: np.ndarray
    label: str               # Class label
    source_file: str         # Origin file
    constellation: str       # "gps" or "galileo"
    scenario: str            # "ds1", "ds2", etc.
    start_sample: int        # Position in original file
    is_spoofed: bool         # Whether this segment is in the spoofed region


def segment_recording(signal: np.ndarray, scenario: OakbatScenario,
                      fs: float = OAKBAT_FS,
                      window_duration_s: float = 0.02,
                      overlap: float = 0.0,
                      clean_label: str = "clean") -> list[Segment]:
    """
    Segment a recording into fixed-length windows, labeling each as
    clean or the appropriate spoofing class based on onset time.

    Args:
        signal:             Complex IQ array
        scenario:           OakbatScenario metadata
        fs:                 Sampling frequency
        window_duration_s:  Window length in seconds (default: 20 ms)
        overlap:            Fractional overlap between windows (0.0 = no overlap)
        clean_label:        Label for pre-onset segments

    Returns:
        List of Segment objects
    """
    window_samples = int(window_duration_s * fs)
    hop_samples = int(window_samples * (1 - overlap))
    onset_sample = int(scenario.onset_seconds * fs)

    segments = []
    start = 0

    while start + window_samples <= len(signal):
        end = start + window_samples
        chunk = signal[start:end]

        # Determine label: clean if entirely before onset, spoofed if after
        if end <= onset_sample:
            label = clean_label
            is_spoofed = False
        elif start >= onset_sample:
            label = scenario.spoofing_class
            is_spoofed = True
        else:
            # Segment straddles onset — skip to avoid ambiguous labels
            start += hop_samples
            continue

        segments.append(Segment(
            data=chunk,
            label=label,
            source_file=scenario.filename,
            constellation=scenario.constellation,
            scenario=scenario.scenario,
            start_sample=start,
            is_spoofed=is_spoofed,
        ))

        start += hop_samples

    return segments


# ---------------------------------------------------------------------------
# Step 3 & 4: Compute STFT spectrogram and resize
# ---------------------------------------------------------------------------
@dataclass
class STFTParams:
    """STFT configuration."""
    nperseg: int = 256        # FFT size / window length
    noverlap: int = 192       # 75% overlap (nperseg * 0.75)
    window: str = "hann"      # Window function
    output_size: tuple = (128, 128)  # Final image dimensions
    log_scale: bool = True    # Use log-magnitude
    epsilon: float = 1e-10    # Floor for log to avoid -inf


def compute_spectrogram(segment_data: np.ndarray, fs: float,
                        params: STFTParams = STFTParams()) -> np.ndarray:
    """
    Compute an STFT spectrogram from an IQ segment and resize it.

    Returns:
        2D float32 array of shape params.output_size
    """
    _, _, Zxx = stft(segment_data, fs=fs,
                     nperseg=params.nperseg,
                     noverlap=params.noverlap,
                     window=params.window)

    # Magnitude spectrogram
    mag = np.abs(Zxx)

    if params.log_scale:
        mag = 10 * np.log10(mag ** 2 + params.epsilon)

    # Resize to target dimensions
    spectrogram = resize(mag, params.output_size,
                         anti_aliasing=True,
                         preserve_range=True).astype(np.float32)

    return spectrogram


# ---------------------------------------------------------------------------
# Step 5: Normalization
# ---------------------------------------------------------------------------
class SpectrogramNormalizer:
    """
    Handles per-image or global (z-score) normalization.

    Usage:
        # Pass 1: fit on training spectrograms
        normalizer = SpectrogramNormalizer(mode="global")
        for spec in training_specs:
            normalizer.update(spec)
        normalizer.finalize()

        # Pass 2: transform all spectrograms
        normalized = normalizer.transform(spec)
    """

    def __init__(self, mode: str = "global"):
        """
        Args:
            mode: "global" for z-score using training set stats,
                  "per_image" for min-max [0,1] per image
        """
        self.mode = mode
        self._running_sum = 0.0
        self._running_sq_sum = 0.0
        self._count = 0
        self.mean = None
        self.std = None

    def update(self, spectrogram: np.ndarray):
        """Accumulate stats for global normalization (call on training set only)."""
        self._running_sum += spectrogram.sum()
        self._running_sq_sum += (spectrogram ** 2).sum()
        self._count += spectrogram.size

    def finalize(self):
        """Compute global mean and std from accumulated stats."""
        if self._count == 0:
            raise RuntimeError("No data accumulated. Call update() first.")
        self.mean = self._running_sum / self._count
        self.std = np.sqrt(self._running_sq_sum / self._count - self.mean ** 2)
        if self.std < 1e-8:
            self.std = 1.0
        print(f"[Normalizer] Global mean={self.mean:.4f}, std={self.std:.4f}")

    def transform(self, spectrogram: np.ndarray) -> np.ndarray:
        """Normalize a spectrogram."""
        if self.mode == "per_image":
            smin, smax = spectrogram.min(), spectrogram.max()
            if smax - smin < 1e-8:
                return np.zeros_like(spectrogram)
            return (spectrogram - smin) / (smax - smin)
        else:  # global z-score
            if self.mean is None:
                raise RuntimeError("Call finalize() before transform().")
            return (spectrogram - self.mean) / self.std

    def save(self, path: str):
        """Save normalization stats for reproducibility."""
        stats = {"mode": self.mode, "mean": float(self.mean), "std": float(self.std)}
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SpectrogramNormalizer":
        """Load previously saved normalization stats."""
        with open(path) as f:
            stats = json.load(f)
        norm = cls(mode=stats["mode"])
        norm.mean = stats["mean"]
        norm.std = stats["std"]
        return norm


# ---------------------------------------------------------------------------
# Step 6: Balanced split creation
# ---------------------------------------------------------------------------
def create_splits(segments: list[Segment],
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  balance_classes: bool = True,
                  max_per_class: Optional[int] = None,
                  seed: int = 42) -> dict[str, list[Segment]]:
    """
    Create stratified train/val/test splits.

    If balance_classes is True, undersamples the majority class(es) to match
    the smallest class. Optionally caps at max_per_class samples per class.
    """
    rng = np.random.default_rng(seed)

    # Group by label
    by_label: dict[str, list[Segment]] = {}
    for seg in segments:
        by_label.setdefault(seg.label, []).append(seg)

    # Print class distribution
    print("\n[Split] Class distribution before balancing:")
    for label, segs in sorted(by_label.items()):
        print(f"  {label}: {len(segs)} segments")

    # Balance
    if balance_classes:
        min_count = min(len(v) for v in by_label.values())
        if max_per_class is not None:
            min_count = min(min_count, max_per_class)
        for label in by_label:
            indices = rng.permutation(len(by_label[label]))[:min_count]
            by_label[label] = [by_label[label][i] for i in indices]
        print(f"[Split] Balanced to {min_count} samples per class")

    # Stratified split
    splits = {"train": [], "val": [], "test": []}
    for label, segs in by_label.items():
        perm = rng.permutation(len(segs))
        n = len(segs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

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


# ---------------------------------------------------------------------------
# Step 7: Save to disk
# ---------------------------------------------------------------------------
def save_dataset(splits: dict[str, list[Segment]],
                 output_dir: str,
                 fs: float,
                 stft_params: STFTParams,
                 normalizer: SpectrogramNormalizer,
                 save_format: str = "npy"):
    """
    Compute spectrograms and save organized by split/class.

    Directory structure:
        output_dir/
            train/
                clean/
                    spec_00001.npy
                spoof_overpowered_instant/
                    spec_00001.npy
                ...
            val/
                ...
            test/
                ...
            metadata.json
            normalization_stats.json
    """
    out = Path(output_dir)
    metadata = {
        "fs": fs,
        "stft_params": {
            "nperseg": stft_params.nperseg,
            "noverlap": stft_params.noverlap,
            "window": stft_params.window,
            "output_size": list(stft_params.output_size),
            "log_scale": stft_params.log_scale,
        },
        "splits": {},
    }

    for split_name, segments in splits.items():
        split_dir = out / split_name
        counters: dict[str, int] = {}
        metadata["splits"][split_name] = {}

        for seg in segments:
            # Compute spectrogram
            spec = compute_spectrogram(seg.data, fs, stft_params)
            spec = normalizer.transform(spec)

            # Save
            label_dir = split_dir / seg.label
            label_dir.mkdir(parents=True, exist_ok=True)

            idx = counters.get(seg.label, 0) + 1
            counters[seg.label] = idx

            if save_format == "npy":
                filepath = label_dir / f"spec_{idx:05d}.npy"
                np.save(str(filepath), spec)
            elif save_format == "png":
                from PIL import Image
                # Scale to 0-255 for image saving
                img_data = ((spec - spec.min()) / (spec.max() - spec.min() + 1e-10) * 255).astype(np.uint8)
                filepath = label_dir / f"spec_{idx:05d}.png"
                Image.fromarray(img_data).save(str(filepath))

        for label, count in counters.items():
            metadata["splits"][split_name][label] = count

    # Save metadata
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    normalizer.save(str(out / "normalization_stats.json"))

    print(f"\n[Save] Dataset saved to {out}")
    print(f"[Save] Metadata written to {out / 'metadata.json'}")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_oakbat_pipeline(oakbat_dir: str, output_dir: str,
                 constellations: list[str] = ["gps", "galileo"],
                 window_duration_s: float = 0.02,
                 overlap: float = 0.0,
                 stft_params: STFTParams = STFTParams(),
                 normalization_mode: str = "global",
                 balance: bool = True,
                 max_per_class: Optional[int] = None,
                 save_format: str = "npy",
                 max_file_duration_s: Optional[float] = None):
    """
    Run the complete OAKBAT processing pipeline.

    Args:
        oakbat_dir:           Directory containing OAKBAT .bin files
        output_dir:           Where to save processed spectrograms
        constellations:       Which constellations to process ["gps", "galileo"]
        window_duration_s:    Segment duration in seconds
        overlap:              Fractional overlap between segments
        stft_params:          STFT configuration
        normalization_mode:   "global" or "per_image"
        balance:              Whether to balance classes
        max_per_class:        Cap on samples per class (None = no cap)
        save_format:          "npy" or "png"
        max_file_duration_s:  Limit on how much of each file to process
    """
    oakbat_path = Path(oakbat_dir)

    # Filter scenarios by requested constellations
    scenarios = [s for s in ALL_SCENARIOS if s.constellation in constellations]

    # -----------------------------------------------------------------------
    # Pass 1: Load and segment all recordings
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("OAKBAT Processing Pipeline")
    print("=" * 60)
    print(f"Source:         {oakbat_dir}")
    print(f"Constellations: {constellations}")
    print(f"Window:         {window_duration_s * 1000:.0f} ms")
    print(f"STFT:           nperseg={stft_params.nperseg}, "
          f"noverlap={stft_params.noverlap}, window={stft_params.window}")
    print(f"Output size:    {stft_params.output_size}")
    print(f"Normalization:  {normalization_mode}")
    print()

    all_segments: list[Segment] = []

    for scenario in scenarios:
        filepath = oakbat_path / scenario.filename

        # ── Try common filename patterns if the expected name isn't found ──
        if not filepath.exists():
            # OAKBAT files may have different naming conventions
            # Try to find a matching file
            pattern = f"*{scenario.constellation}*{scenario.scenario}*"
            candidates = list(oakbat_path.glob(pattern))
            if candidates:
                filepath = candidates[0]
                print(f"  [Note] Using matched file: {filepath.name}")
            else:
                print(f"  [SKIP] File not found: {scenario.filename} "
                      f"(no match for {pattern})")
                continue

        print(f"  Loading {filepath.name} ({scenario.constellation} "
              f"{scenario.scenario}: {scenario.description})...")

        signal = load_oakbat_iq(str(filepath), OAKBAT_FS, max_file_duration_s)
        print(f"    → {len(signal)} samples ({len(signal) / OAKBAT_FS:.1f} s)")

        segments = segment_recording(signal, scenario, OAKBAT_FS,
                                     window_duration_s, overlap)
        print(f"    → {len(segments)} segments "
              f"(clean: {sum(1 for s in segments if not s.is_spoofed)}, "
              f"spoofed: {sum(1 for s in segments if s.is_spoofed)})")

        all_segments.extend(segments)

    if not all_segments:
        print("\n[ERROR] No segments generated. Check file paths and names.")
        return

    print(f"\nTotal segments: {len(all_segments)}")

    # -----------------------------------------------------------------------
    # Pass 2: Create balanced splits
    # -----------------------------------------------------------------------
    splits = create_splits(all_segments, balance_classes=balance,
                           max_per_class=max_per_class)

    # -----------------------------------------------------------------------
    # Pass 3: Fit normalizer on training spectrograms
    # -----------------------------------------------------------------------
    normalizer = SpectrogramNormalizer(mode=normalization_mode)

    if normalization_mode == "global":
        print("\n[Norm] Computing global stats from training set...")
        for i, seg in enumerate(splits["train"]):
            spec = compute_spectrogram(seg.data, OAKBAT_FS, stft_params)
            normalizer.update(spec)
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(splits['train'])} training specs")
        normalizer.finalize()

    # -----------------------------------------------------------------------
    # Pass 4: Save all splits
    # -----------------------------------------------------------------------
    save_dataset(splits, output_dir, OAKBAT_FS, stft_params,
                 normalizer, save_format)

    print("\n" + "=" * 60)
    print("OAKTABT Pipeline complete!")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OAKBAT → STFT Spectrogram Processing Pipeline")
    parser.add_argument("--oakbat-dir", type=str, required=True,
                        help="Directory containing OAKBAT .bin files")
    parser.add_argument("--output-dir", type=str, default="./oakbat_spectrograms",
                        help="Output directory for processed spectrograms")
    parser.add_argument("--constellations", nargs="+",
                        choices=["gps", "galileo"], default=["gps", "galileo"],
                        help="Which constellations to process")
    parser.add_argument("--window-ms", type=float, default=20.0,
                        help="Segment window duration in milliseconds")
    parser.add_argument("--overlap", type=float, default=0.0,
                        help="Fractional overlap between windows (0.0-0.9)")
    parser.add_argument("--fft-size", type=int, default=256,
                        help="STFT FFT size / window length")
    parser.add_argument("--fft-overlap-frac", type=float, default=0.75,
                        help="STFT overlap as fraction of FFT size")
    parser.add_argument("--output-size", type=int, default=128,
                        help="Spectrogram output dimension (NxN)")
    parser.add_argument("--normalization", choices=["global", "per_image"],
                        default="global",
                        help="Normalization strategy")
    parser.add_argument("--no-balance", action="store_true",
                        help="Skip class balancing")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Maximum samples per class (for quick experiments)")
    parser.add_argument("--format", choices=["npy", "png"], default="npy",
                        help="Output format for spectrograms")
    parser.add_argument("--max-duration", type=float, default=None,
                        help="Max seconds to load per file (for testing)")

    args = parser.parse_args()

    stft_params = STFTParams(
        nperseg=args.fft_size,
        noverlap=int(args.fft_size * args.fft_overlap_frac),
        output_size=(args.output_size, args.output_size),
    )

    run_oakbat_pipeline(
        oakbat_dir=args.oakbat_dir,
        output_dir=args.output_dir,
        constellations=args.constellations,
        window_duration_s=args.window_ms / 1000,
        overlap=args.overlap,
        stft_params=stft_params,
        normalization_mode=args.normalization,
        balance=not args.no_balance,
        max_per_class=args.max_per_class,
        save_format=args.format,
        max_file_duration_s=args.max_duration,
    )
