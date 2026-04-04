"""
Dataset processing module for all datasets currently in use. Each dataset is defined 
in its own dedicated directory, as well as the function used to process them.

Processing is done in 2 steps:
Pass 1: Stream files -> Compute spectrograms -> Accumulate normalizer stats
         (one file at a time, IQ discarded after spectrogram computation)
Pass 2: Stream files again -> Compute spectrograms -> normalize -> Save to disk

Additionally, there is a feature extraction process done for improving results 
in the case of spoofing, the features are assembled as a vector of floating-point
values, those being:
    1. Mean_power          — mean |IQ|², proxy for received signal level
    2. PAPR                — peak-to-average power ratio, sensitive to overpowered spoofing
    3. Spectral_kurtosis   — kurtosis of PSD; deviates from [3] under jamming/spoofing
    4. Spectral_skewness   — skewness of PSD; asymmetric energy distribution
    5. Spectral_flatness   — geometric/arithmetic mean of PSD; low = structured signal
    6. Spectral_entropy    — Shannon entropy of normalised PSD; drops under coherent RFI
    7. Inst_bandwidth      — 90%-power spectral bandwidth as fraction of Nyquist
    8. Caf_peak_ratio      — normalised secondary autocorrelation peak; elevated when
                              a second signal overlaps (spoofing indicator)

Literature support:
    - Kurtosis, flatness, entropy, bandwidth:
        van der Merwe et al. (2023), Sensors 23(7):3452
        Rijnsdorp et al. (2023), Eng. Proc. 54:60
    - Skewness as spectral classifier feature:
        XAI GNSS, Sensors 24(24):8039 (2024)
        Contreras Franco et al. (2024), IEEE TAES 60(3):2705
    - CAF / autocorrelation for spoofing detection:
        Borhani-Darian et al. (2024), EURASIP J. Adv. Signal Process. 2024:14
    - Multi-feature CNN fusion architecture:
        Ebrahimi Mehr & Dovis (2025), IEEE TAES 61(2):1660
"""

from datasets.OakbatSpoofing import *
from datasets.SwinneyJamming import *
from feature_extraction import *
from dataclasses import dataclass, field
import json
import numpy as np
from scipy.signal import stft
from skimage.transform import resize

# Number of features produced by compute_features()
N_FEATURES = 8

# Sample rate
SAMPLE_RATE = 5e6  # 5 MHz

@dataclass
class Segment:
    data: np.ndarray          # Complex IQ samples (complex64)
    label: str                # Class label
    source_file: str          # Origin filename
    scenario: str             # Scenario identifier
    start_sample: int         # Offset within the source file
    is_spoofed: bool          # True for post-onset OAKBAT windows
    features: Optional[np.ndarray] = field(default=None)
    # Shape: (N_FEATURES,) float32, populated by compute_features().
    # None means features have not been computed yet — existing code that
    # does not use features is unaffected.

@dataclass
class STFTParams:
    """STFT configuration — must match between training and inference."""
    nperseg:     int   = 256
    noverlap:    int   = 192        # 75% overlap
    window:      str   = "hann"     # Hann windowing
    output_size: tuple = (128, 128)
    log_scale:   bool  = True
    epsilon:     float = 1e-10


def compute_spectrogram(segment_data: np.ndarray, fs: float,
                        params: STFTParams = STFTParams()) -> np.ndarray:
    """
    Compute a log-magnitude STFT spectrogram from an IQ segment and resize
    it to params.output_size. Returns float32 array of shape output_size.
    Unchanged from previous version.
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


class SpectrogramNormalizer:
    """Handles per-image or global (z-score) normalization."""

    def __init__(self, mode: str = "global"):
        self.mode = mode
        self._running_sum = 0.0
        self._running_sq_sum = 0.0
        self._count = 0
        self.mean = None
        self.std = None

    def update(self, spectrogram: np.ndarray):
        self._running_sum += spectrogram.sum()
        self._running_sq_sum += (spectrogram ** 2).sum()
        self._count += spectrogram.size

    def finalize(self):
        if self._count == 0:
            raise RuntimeError("No data accumulated. Call update() first.")
        self.mean = self._running_sum / self._count
        self.std = np.sqrt(self._running_sq_sum / self._count - self.mean ** 2)
        if self.std < 1e-8:
            self.std = 1.0
        print(f"[Normalizer] Global mean={self.mean:.4f}, std={self.std:.4f}")

    def transform(self, spectrogram: np.ndarray) -> np.ndarray:
        if self.mode == "per_image":
            smin, smax = spectrogram.min(), spectrogram.max()
            if smax - smin < 1e-8:
                return np.zeros_like(spectrogram)
            return (spectrogram - smin) / (smax - smin)
        else:
            if self.mean is None:
                raise RuntimeError("Call finalize() before transform().")
            return (spectrogram - self.mean) / self.std

    def save(self, path: str):
        stats = {"mode": self.mode, "mean": float(self.mean), "std": float(self.std)}
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SpectrogramNormalizer":
        with open(path) as f:
            stats = json.load(f)
        norm = cls(mode=stats["mode"])
        norm.mean = stats["mean"]
        norm.std = stats["std"]
        return norm


def create_splits(segments: list[Segment],
                  train_ratio: float = 0.7,
                  val_ratio:   float = 0.15,
                  balance_classes: bool = True,
                  max_per_class:   Optional[int] = None,
                  seed: int = 42) -> dict[str, list[Segment]]:
    """
    Create stratified train / val / test splits with optional class balancing.
    Unchanged from previous version.
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