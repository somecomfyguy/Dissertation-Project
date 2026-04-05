
from common.types import STFTParams
import json
import numpy as np
from scipy.signal import stft
from skimage.transform import resize
from typing import Optional


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