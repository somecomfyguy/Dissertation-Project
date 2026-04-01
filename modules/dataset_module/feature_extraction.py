"""
Feature extraction helper functions
"""
import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
import json
from typing import Optional

# Sample rate
SAMPLE_RATE = 5e6  # 5 MHz

# Internal helpers: PSD and CAF Ratio
def _psd(iq: np.ndarray) -> np.ndarray:
    """
    Power spectral density estimate via periodogram (no windowing — fast and
    consistent across segments of fixed length). Returns a 1-D float64 array
    of length len(iq)//2 + 1 (one-sided for real analysis; we use the full
    complex spectrum here for complex IQ).
    """
    N   = len(iq)
    fft = np.fft.fft(iq, n=N)
    return (np.abs(fft) ** 2) / N        # shape: (N,), unnormalised periodogram


def _caf_peak_ratio(iq: np.ndarray) -> float:
    """
    Lightweight CAF proxy via the normalised autocorrelation function (ACF).

    Computation (Wiener-Khinchin theorem):
        ACF = IFFT( |FFT(IQ)|^2 )

    For a noise-like clean GNSS signal, ACF[τ>0] ≈ 0.
    For a spoofed segment where a second code-aligned signal overlaps the
    legitimate one, ACF shows a secondary peak at the code-phase offset
    between the two signals — the same feature exploited by the full CAF
    in Borhani-Darian et al. (2024, EURASIP).

    Returns the maximum of |ACF_normalised[1 : N//4]|, i.e. the largest
    secondary peak relative to the zero-lag peak. Values near 0 indicate
    a single signal; elevated values indicate signal overlap.

    Complexity: O(N log N) — two FFTs of length N.
    """
    N = len(iq)
    p = _psd(iq)                             # |FFT|^2 / N
    acf = np.fft.ifft(p).real               # ACF via Wiener-Khinchin
    zero_lag = acf[0]
    if zero_lag < 1e-12:
        return 0.0
    acf_norm = np.abs(acf / zero_lag)
    # Search lags 1 … N//4 (GPS C/A code period = 1 ms = 5000 samples at
    # 5 MHz; a 20 ms window contains 20 code periods so N//4 = 25 000 gives
    # ample range to capture any plausible spoofing code-phase offset).
    return float(np.max(acf_norm[1: N // 4]))


def compute_features(iq: np.ndarray,
                     fs: float = SAMPLE_RATE,
                     epsilon: float = 1e-12) -> np.ndarray:
    """
    Extract an 8-element feature vector from a raw IQ segment.
    All features are computable without a software GNSS receiver.

    Args:
        iq:       Complex IQ samples (complex64 or complex128)
        fs:       Sampling frequency in Hz (used for bandwidth calculation)
        epsilon:  Floor added before log/division to avoid numerical issues

    Returns:
        float32 array of shape (N_FEATURES,) = (8,)

    Feature descriptions and literature support
    -------------------------------------------
    [0] mean_power
        Mean instantaneous power E[|x|²]. A spoofed signal's total power
        is elevated relative to a clean signal baseline.
        Ref: Tanış & Cetin (2024), Int. J. Circuit Theory Appl.

    [1] papr
        Peak-to-average power ratio: max(|x|²) / mean(|x|²).
        Captures burst-like power structure. Overpowered spoofing (OAKBAT
        ds1/ds2) produces distinct PAPR signatures.
        Ref: implicit in XAI GNSS (Sensors 24(24):8039, 2024).

    [2] spectral_kurtosis
        Kurtosis of the PSD distribution. Clean GNSS ≈ Gaussian (kurtosis
        ≈ 3). Pulse jamming and spoofing onset cause sharp deviations.
        Ref: van der Merwe et al. (2023), Sensors 23(7):3452;
             Rijnsdorp et al. (2023), Eng. Proc. 54:60.

    [3] spectral_skewness
        Skewness of the PSD distribution. Asymmetric spectral energy
        (e.g. single-sided AM/FM sidebands) produces non-zero skewness.
        Ref: Contreras Franco et al. (2024), IEEE TAES 60(3):2705;
             XAI GNSS, Sensors 24(24):8039.

    [4] spectral_flatness
        Geometric mean / arithmetic mean of the PSD (Wiener entropy).
        Near 1 for white noise; near 0 for tonal/structured signals.
        Spectral flatness is the strongest single discriminator of clean
        vs. interfered signals per SHAP analysis.
        Ref: XAI GNSS (Sensors 24(24):8039, 2024).

    [5] spectral_entropy
        Normalised Shannon entropy of the PSD treated as a probability
        distribution. Maximised for noise-like signals; drops sharply when
        coherent RFI concentrates energy in narrow spectral regions.
        Ref: van der Merwe et al. (2023), Sensors 23(7):3452.

    [6] inst_bandwidth
        90%-power spectral bandwidth, normalised to [0, 1] by the Nyquist
        frequency (fs/2). Distinguishes narrowband CW/NB jammers from
        wideband chirp/FM jammers even at low JSR.
        Ref: Rijnsdorp et al. (2023), Eng. Proc. 54:60.

    [7] caf_peak_ratio
        Normalised secondary autocorrelation peak (see _caf_peak_ratio).
        Elevated values indicate a second signal overlapping the primary —
        the key fingerprint of spoofing signal overlay.
        Ref: Borhani-Darian et al. (2024), EURASIP J. Adv. Signal Process.
             doi:10.1186/s13634-023-01103-1
    """
    iq = np.asarray(iq, dtype=np.complex64)
    N  = len(iq)

    # ---- Time-domain features ------------------------------------------ #
    power     = np.abs(iq) ** 2                       # instantaneous power
    mean_pow  = float(np.mean(power))
    papr      = float(np.max(power) / (mean_pow + epsilon))

    # ---- Spectral features --------------------------------------------- #
    p = _psd(iq)                                      # shape (N,), raw periodogram

    # Normalise PSD to a probability distribution for entropy/moments
    p_sum = p.sum() + epsilon
    p_norm = p / p_sum                               # sums to ~1

    # Kurtosis and skewness of the PSD distribution
    # Using scipy for numerical stability (Fisher definition: excess kurtosis)
    spec_kurt = float(scipy_kurtosis(p, fisher=True, bias=False))
    spec_skew = float(scipy_skew(p, bias=False))

    # Spectral flatness (geometric mean / arithmetic mean)
    log_p      = np.log(p + epsilon)
    geom_mean  = float(np.exp(np.mean(log_p)))
    arith_mean = float(np.mean(p) + epsilon)
    flatness   = geom_mean / arith_mean              # ∈ [0, 1]

    # Spectral entropy (normalised Shannon entropy)
    entropy = float(-np.sum(p_norm * np.log2(p_norm + epsilon)))
    max_entropy = np.log2(N)
    entropy_norm = entropy / max_entropy             # ∈ [0, 1]

    # Instantaneous bandwidth (90% power containment), normalised to [0,1]
    cumulative   = np.cumsum(p_norm)
    low_idx      = int(np.searchsorted(cumulative, 0.05))
    high_idx     = int(np.searchsorted(cumulative, 0.95))
    bw_bins      = high_idx - low_idx                # in FFT bins
    bw_norm      = float(bw_bins) / N               # fraction of total bins

    # ---- CAF proxy (autocorrelation secondary peak) --------------------- #
    caf_ratio = _caf_peak_ratio(iq)

    return np.array([
        mean_pow,
        papr,
        spec_kurt,
        spec_skew,
        flatness,
        entropy_norm,
        bw_norm,
        caf_ratio,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Feature normalizer (mirrors structure of SpectrogramNormalizer)
# ---------------------------------------------------------------------------
class FeatureNormalizer:
    """
    Per-feature z-score normalizer for the 8-element feature vector.

    Usage (same two-pass pattern as SpectrogramNormalizer):
        # Pass 1: accumulate training set statistics
        norm = FeatureNormalizer()
        for seg in train_segments:
            norm.update(seg.features)
        norm.finalize()

        # Pass 2: transform all features
        feat_norm = norm.transform(feat)

    Save/load to JSON to keep feature stats alongside spectrogram stats.
    """

    def __init__(self):
        self._sum    = np.zeros(N_FEATURES, dtype=np.float64)
        self._sq_sum = np.zeros(N_FEATURES, dtype=np.float64)
        self._count  = 0
        self.mean: Optional[np.ndarray] = None
        self.std:  Optional[np.ndarray] = None

    def update(self, features: np.ndarray):
        """Accumulate statistics from one feature vector."""
        f = np.asarray(features, dtype=np.float64)
        self._sum    += f
        self._sq_sum += f ** 2
        self._count  += 1

    def finalize(self):
        """Compute per-feature mean and std from accumulated statistics."""
        if self._count == 0:
            raise RuntimeError("No data accumulated. Call update() first.")
        self.mean = self._sum / self._count
        variance  = self._sq_sum / self._count - self.mean ** 2
        self.std  = np.sqrt(np.maximum(variance, 1e-12))
        print(f"[FeatureNormalizer] mean={self.mean.round(4)}")
        print(f"[FeatureNormalizer]  std={self.std.round(4)}")

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Z-score normalise a feature vector using fitted statistics."""
        if self.mean is None:
            raise RuntimeError("Call finalize() before transform().")
        return ((np.asarray(features, dtype=np.float32) - self.mean)
                / self.std).astype(np.float32)

    def save(self, path: str):
        """Save normalizer statistics to a JSON file."""
        stats = {
            "mean": self.mean.tolist(),
            "std":  self.std.tolist(),
        }
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[FeatureNormalizer] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FeatureNormalizer":
        """Load previously saved normalizer statistics."""
        with open(path) as f:
            stats = json.load(f)
        norm      = cls()
        norm.mean = np.array(stats["mean"], dtype=np.float32)
        norm.std  = np.array(stats["std"],  dtype=np.float32)
        return norm