"""
Spectrogram computation and normalization utilities.
"""
import json
from typing import Optional

import numpy as np
from scipy.signal import stft
from skimage.transform import resize

from modules.common.types import STFTParams


# Spectrogram computation
def compute_spectrogram(segment_data: np.ndarray,
                        fs: float,
                        params: STFTParams = STFTParams()) -> np.ndarray:
    """
    Compute a log-magnitude STFT spectrogram from an IQ segment.

    Args:
        segment_data: Complex IQ samples of shape (num_samples,).
        fs:           Sampling frequency in Hz.
        params:       STFT configuration.

    Returns:
        Float32 array of shape params.output_size (default 128x128).
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
    Global z-score or per-image min-max normalizer for spectrograms.\n
    For global mode, statistics are accumulated from the training split\n
    only (to avoid data leakage), then applied uniformly to all splits.
    """

    def __init__(self, mode: str = "global"):
        self.mode             = mode
        self._running_sum     = 0.0
        self._running_sq_sum  = 0.0
        self._count           = 0
        self.mean: Optional[float] = None
        self.std:  Optional[float] = None

    def update(self, spectrogram: np.ndarray) -> None:
        """Accumulate pixel-level statistics. Call on training set only."""
        self._running_sum    += spectrogram.sum()
        self._running_sq_sum += (spectrogram ** 2).sum()
        self._count          += spectrogram.size

    def finalize(self) -> None:
        """Compute global mean and std from accumulated statistics."""
        if self._count == 0:
            raise RuntimeError("No data accumulated. Call update() first.")
        self.mean = self._running_sum / self._count
        self.std  = np.sqrt(self._running_sq_sum / self._count - self.mean ** 2)
        if self.std < 1e-8:
            self.std = 1.0
        print(f"[Normalizer] Global mean={self.mean:.4f}, std={self.std:.4f}")

    def transform(self, spectrogram: np.ndarray) -> np.ndarray:
        """Normalize a spectrogram using the fitted or per-image strategy."""
        if self.mode == "global":
            return ((spectrogram - self.mean) / self.std).astype(np.float32)
        else:
            lo, hi = spectrogram.min(), spectrogram.max()
            return ((spectrogram - lo) / (hi - lo + 1e-10)).astype(np.float32)

    def save(self, path: str) -> None:
        """Persist normalizer statistics to JSON."""
        stats = {"mode": self.mode, "mean": float(self.mean), "std": float(self.std)}
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SpectrogramNormalizer":
        """Restore a SpectrogramNormalizer from a saved JSON file."""
        with open(path) as f:
            stats = json.load(f)
        norm      = cls(mode=stats["mode"])
        norm.mean = stats["mean"]
        norm.std  = stats["std"]
        return norm