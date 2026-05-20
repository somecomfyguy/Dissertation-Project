"""
Spectrogram augmentation transforms for domain-robust training.

Each transform simulates a specific domain shift observed in the
cross-dataset evaluation:
    - PowerOffset:     ±dB shift in global power level
                       (TEXBAT/GATEMAN vs OAKBAT amplitude mismatch)
    - SpectralTilt:    linear gradient across frequency axis
                       (front-end filter roll-off differences)
    - GaussianNoise:   additive noise at variable SNR
                       (receiver noise floor variation)
    - FrequencyMask:   zero out random frequency bands
                       (SpecAugment-style, regularisation)
    - TimeMask:        zero out random time columns
                       (SpecAugment-style, regularisation)

All transforms operate on single-channel float32 spectrograms of shape
(H, W) and are applied stochastically during training only.
"""
import numpy as np


class SpectrogramAugmentor:
    """
    Applies a random subset of augmentations to each spectrogram.

    Usage:
        aug = SpectrogramAugmentor()
        spec = aug(spec)  # during training only
    """

    def __init__(self,
                 power_offset_db: float = 5.0,
                 power_offset_prob: float = 0.5,
                 tilt_max: float = 0.3,
                 tilt_prob: float = 0.3,
                 noise_std_range: tuple = (0.01, 0.1),
                 noise_prob: float = 0.5,
                 freq_mask_max: int = 15,
                 freq_mask_prob: float = 0.3,
                 time_mask_max: int = 15,
                 time_mask_prob: float = 0.3,
                 seed: int = None):
        """
        Args:
            power_offset_db:   Max absolute power offset in dB.
            power_offset_prob: Probability of applying power offset.
            tilt_max:          Max slope of spectral tilt (per row).
            tilt_prob:         Probability of applying spectral tilt.
            noise_std_range:   (min, max) std of additive Gaussian noise.
            noise_prob:        Probability of adding noise.
            freq_mask_max:     Max number of frequency bins to mask.
            freq_mask_prob:    Probability of frequency masking.
            time_mask_max:     Max number of time bins to mask.
            time_mask_prob:    Probability of time masking.
            seed:              Optional RNG seed for reproducibility.
        """
        self.power_offset_db = power_offset_db
        self.power_offset_prob = power_offset_prob
        self.tilt_max = tilt_max
        self.tilt_prob = tilt_prob
        self.noise_std_range = noise_std_range
        self.noise_prob = noise_prob
        self.freq_mask_max = freq_mask_max
        self.freq_mask_prob = freq_mask_prob
        self.time_mask_max = time_mask_max
        self.time_mask_prob = time_mask_prob
        self.seed = seed

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to a spectrogram.

        Args:
            spec: float32 array of shape (H, W).

        Returns:
            Augmented float32 array of same shape.
        """
        if spec is None:
            return spec
        spec = spec.copy()
        H, W = spec.shape
        rng = np.random.default_rng()

        # Power offset: shift all pixels by a random dB value.
        # In normalised spectrogram space, 1 dB ≈ 1/std units,
        # so we scale by a fraction of the spectrogram's range.
        if rng.random() < self.power_offset_prob:
            offset = rng.uniform(-self.power_offset_db,
                                       self.power_offset_db)
            spec_range = spec.max() - spec.min()
            # Scale offset relative to spectrogram dynamic range
            spec += offset * spec_range / 20.0

        # Spectral tilt: linear gradient across frequency (rows)
        if rng.random() < self.tilt_prob:
            slope = rng.uniform(-self.tilt_max, self.tilt_max)
            tilt = np.linspace(-slope, slope, H).reshape(-1, 1)
            spec += tilt

        # Additive Gaussian noise
        if rng.random() < self.noise_prob:
            std = rng.uniform(*self.noise_std_range)
            noise = rng.normal(0, std, size=spec.shape)
            spec += noise.astype(np.float32)

        # Frequency masking (SpecAugment)
        if rng.random() < self.freq_mask_prob:
            mask_width = rng.integers(1, self.freq_mask_max + 1)
            start = rng.integers(0, max(1, H - mask_width))
            spec[start:start + mask_width, :] = spec.mean()

        # Time masking (SpecAugment)
        if rng.random() < self.time_mask_prob:
            mask_width = rng.integers(1, self.time_mask_max + 1)
            start = rng.integers(0, max(1, W - mask_width))
            spec[:, start:start + mask_width] = spec.mean()

        return spec.astype(np.float32)