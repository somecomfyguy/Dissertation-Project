"""
Statistical feature extraction from raw IQ segments.

Produces an 8-element float32 feature vector from a complex IQ window
without requiring a software GNSS receiver (no correlation, no tracking
loops). All features operate directly on the raw baseband samples and
are therefore applicable to both GPS L1 and Galileo E1 signals.

Feature vector layout (indices match compute_features() return order):
    [0] mean_power        — mean |IQ|², proxy for received signal level
    [1] papr              — peak-to-average power ratio
    [2] spectral_kurtosis — kurtosis of PSD; deviates from ~3 under RFI
    [3] spectral_skewness — skewness of PSD; asymmetric sideband energy
    [4] spectral_flatness — geometric/arithmetic mean of PSD (Wiener entropy)
    [5] spectral_entropy  — normalised Shannon entropy of PSD
    [6] inst_bandwidth    — 90%-power bandwidth as fraction of Nyquist
    [7] caf_peak_ratio    — normalised secondary autocorrelation peak
"""
import json
from typing import Optional

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew

from modules.common.types import N_FEATURES, SAMPLE_RATE


# Internal helpers: PSD and CAF Ratio
def _psd(iq: np.ndarray) -> np.ndarray:
    """
    Power spectral density estimate via periodogram, using no windowing. 
    
    Args:
        iq: array of IQ samples

    Returns:
        np.ndarray: float64 array of shape (N, ) and length (len(iq)//2 + 1)\n 
        (one-sided for real analysis; complex spectrum for complex IQ).
    """
    N = len(iq)
    fft = np.fft.fft(iq, n=N)
    return (np.abs(fft) ** 2) / N   # shape: (N,), unnormalised periodogram


def _caf_peak_ratio(iq: np.ndarray) -> float:
    """
    Lightweight CAF proxy via the normalised autocorrelation function (ACF).\n
    Computation (Wiener-Khinchin theorem): ACF = IFFT( |FFT(IQ)|^2 )\n
    The search range [1, N//4] covers lags up to N//4 samples. \n
    Source: Borhani-Darian et al. (2024, EURASIP).

    Args:
        iq: array of IQ samples

    Returns:
        float: the maximum of |ACF_normalised[1 : N//4]|
    """
    N = len(iq)
    p = _psd(iq)                # |FFT|^2 / N
    acf = np.fft.ifft(p).real   # ACF via Wiener-Khinchin
    zero_lag = acf[0]
    if zero_lag < 1e-12:
        return 0.0
    acf_norm = np.abs(acf / zero_lag)
    # Search lags 1 … N//4 (GPS C/A code period = 1 ms = 5000 samples at
    # 5 MHz; a 20 ms window contains 20 code periods, N//4 = 25 000.
    # covers one GPS C/A code period (1 ms = 5 000 samples)
    # several times over.
    return float(np.max(acf_norm[1: N // 4]))


def compute_features(iq: np.ndarray,
                     fs: float = SAMPLE_RATE,
                     epsilon: float = 1e-12) -> np.ndarray:
    """
    Extract an 8-element feature vector from a raw IQ segment. \n
    All features are computable without a software GNSS receiver.

    Args:
        iq:       Complex IQ samples (complex64 or complex128)
        fs:       Sampling frequency in Hz (used for bandwidth calculation)
        epsilon:  Small floor added before logarithms and divisions to prevent \n
         -inf / NaN on silent or near-silent inputs.

    Returns:
        float32 array of shape (N_FEATURES,) = \n
            mean_power,\n
            papr, \n
            spectral_kurtosis, \n
            spectral_skewness,\n
            spectral_flatness, \n
            spectral_entropy, \n
            inst_bandwidth, \n
            caf_peak_ratio\n
    """
    iq = np.asarray(iq, dtype=np.complex64)
    N  = len(iq)

    # [0] mean_power — mean instantaneous power E[|x|²].
    # A spoofed signal's total power is elevated relative to the clean
    # baseline because the spoofer broadcasts on top of the authentic signal.
    # Ref: Tanış & Cetin (2024), Int. J. Circuit Theory Appl.
    power = np.abs(iq) ** 2     # instantaneous power
    mean_pow  = float(np.mean(power))

    # [1] papr — peak-to-average power ratio: max(|x|²) / E[|x|²].
    # Captures burst-like or impulsive power structure. Overpowered spoofing
    # scenarios (OAKBAT ds1/ds2) produce distinct PAPR signatures relative
    # to continuous-wave jamming or clean signals.
    papr = float(np.max(power) / (mean_pow + epsilon))

    # Raw PSD and normalised version used by entropy/flatness/bandwidth.
    p = _psd(iq)                  # shape (N,), unnormalised
    p_norm = p / (p.sum() + epsilon)   # treated as a probability distribution

    # [2] spectral_kurtosis — excess kurtosis of the PSD distribution.
    # A Gaussian noise floor yields kurtosis ≈ 0 (Fisher convention).
    # Pulse jamming and spoofing onset cause sharp spikes, deviating strongly.
    # Ref: van der Merwe et al. (2023), Sensors 23(7):3452;
    #      Rijnsdorp et al. (2023), Eng. Proc. 54:60.
    spec_kurt = float(scipy_kurtosis(p, fisher=True, bias=False))

    # [3] spectral_skewness — skewness of the PSD distribution.
    # Asymmetric spectral energy (e.g. single-sided AM/FM sidebands) produces
    # non-zero skewness that is near-zero for symmetric clean or CW signals.
    # Ref: Contreras Franco et al. (2024), IEEE TAES 60(3):2705;
    #      XAI GNSS, Sensors 24(24):8039.
    spec_skew = float(scipy_skew(p, bias=False))

    # [4] spectral_flatness — Wiener entropy: geometric mean / arithmetic mean
    # of the PSD. Near 1 for white noise; near 0 for tonal/structured signals.
    # SHAP analysis identifies this as the strongest single discriminator of
    # clean vs. interfered signals.
    # Ref: XAI GNSS, Sensors 24(24):8039.
    log_p      = np.log(p + epsilon)
    geom_mean  = float(np.exp(np.mean(log_p)))
    arith_mean = float(np.mean(p) + epsilon)
    flatness   = geom_mean / arith_mean              # ∈ [0, 1]

    # [5] spectral_entropy — normalised Shannon entropy of the PSD treated as
    # a probability distribution. Maximised (≈ 1) for noise-like signals;
    # drops sharply when coherent RFI concentrates energy in narrow bands.
    # Ref: van der Merwe et al. (2023), Sensors 23(7):3452.
    entropy = float(-np.sum(p_norm * np.log2(p_norm + epsilon)))
    entropy_norm = entropy / np.log2(N)    # ∈ [0, 1]

    # [6] inst_bandwidth — 90%-power containment bandwidth, normalised to
    # [0, 1] as a fraction of the total FFT bins. Computed by finding the
    # 5th and 95th percentiles of the cumulative PSD distribution.
    cumulative = np.cumsum(p_norm)
    low_idx    = int(np.searchsorted(cumulative, 0.05))
    high_idx   = int(np.searchsorted(cumulative, 0.95))
    bw_norm    = float(high_idx - low_idx) / N    # fraction of Nyquist

    # [7] caf_peak_ratio — normalised secondary autocorrelation peak.
    # See _caf_peak_ratio() for full description.
    caf_ratio = _caf_peak_ratio(iq)


    result = np.array([mean_pow, papr, spec_kurt, spec_skew,
                       flatness, entropy_norm, bw_norm, caf_ratio],
                      dtype=np.float32)

    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result


class FeatureNormalizer:
    """
    Per-feature z-score normalizer for the 8-element feature vector.

    Typical usage:
        # Pass 1: accumulate training set statistics
        norm = FeatureNormalizer()
        for seg in train_segments:
            norm.update(seg.features)
        norm.finalize()

        # Pass 2: normalize all splits
        feat_norm = norm.transform(feat)

    Save/load to JSON alongside the spectrogram normalization_stats.json
    to keep all preprocessing parameters together.
    """

    def __init__(self):
        self._sum    = np.zeros(N_FEATURES, dtype=np.float64)
        self._sq_sum = np.zeros(N_FEATURES, dtype=np.float64)
        self._count  = 0
        self.mean: Optional[np.ndarray] = None
        self.std:  Optional[np.ndarray] = None

    def update(self, features: np.ndarray) -> None:
        """
        Accumulate per-feature sum and sum-of-squares from one feature vector.

        Must be called once per training segment before finalize().

        Args:
            features: Float array of shape (N_FEATURES,).
        """
        f             = np.asarray(features, dtype=np.float64)
        self._sum    += f
        self._sq_sum += f ** 2
        self._count  += 1

    def finalize(self) -> None:
        """
        Compute per-feature mean and standard deviation from accumulated stats.

        Raises:
            RuntimeError: If update() has not been called at least once.
        """
        if self._count == 0:
            raise RuntimeError("No data accumulated. Call update() first.")
        self.mean    = self._sum / self._count
        variance     = self._sq_sum / self._count - self.mean ** 2
        self.std     = np.sqrt(np.maximum(variance, 1e-12))
        print(f"[FeatureNormalizer] mean={self.mean.round(4)}")
        print(f"[FeatureNormalizer]  std={self.std.round(4)}")

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalisation: (features - mean) / std.

        Args:
            features: Float array of shape (N_FEATURES,).

        Returns:
            Normalised float32 array of shape (N_FEATURES,).

        Raises:
            RuntimeError: If finalize() has not been called.
        """
        if self.mean is None:
            raise RuntimeError("Call finalize() before transform().")
        return ((np.asarray(features, dtype=np.float32) - self.mean)
                / self.std).astype(np.float32)

    def save(self, path: str) -> None:
        """
        Persist normalizer statistics to a JSON file.

        Args:
            path: Destination file path (e.g. 'feature_norm_stats.json').
        """
        stats = {"mean": self.mean.tolist(), "std": self.std.tolist()}
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[FeatureNormalizer] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FeatureNormalizer":
        """
        Restore a FeatureNormalizer from a previously saved JSON file.

        Args:
            path: Path to the JSON file produced by save().

        Returns:
            FeatureNormalizer with mean and std populated, ready for transform().
        """
        with open(path) as f:
            stats = json.load(f)
        norm      = cls()
        norm.mean = np.array(stats["mean"], dtype=np.float32)
        norm.std  = np.array(stats["std"],  dtype=np.float32)
        return norm