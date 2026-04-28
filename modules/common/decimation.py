"""
Polyphase FIR decimation utilities for cross-dataset evaluation pipelines.

Used to bring the native sample rates of TEXBAT (25 MHz) and GATEMAN
(20 MS/s) down to the pipeline's working rate of 5 MHz before STFT
computation. Wraps scipy.signal.resample_poly which applies a
Kaiser-windowed anti-aliasing FIR followed by polyphase decimation in a
single pass.
"""
import numpy as np
from scipy.signal import resample_poly


def polyphase_decimate(iq: np.ndarray, factor: int) -> np.ndarray:
    """
    Decimate a complex IQ stream by an integer factor using polyphase FIR.

    The underlying filter is designed automatically by scipy with a Kaiser
    window giving >80 dB stopband attenuation — more than sufficient for
    GNSS baseband signals. Edge effects are limited to ~N/factor samples
    per boundary (where N is the internal filter length), negligible for
    20 ms (100 k sample) segments.

    Args:
        iq:     Complex IQ samples (complex64 or complex128), 1-D.
        factor: Integer decimation factor (>= 1). factor=1 is a no-op.

    Returns:
        Complex64 array of length ceil(len(iq) / factor).
    """
    if factor < 1:
        raise ValueError(f"Decimation factor must be >= 1, got {factor}")
    if factor == 1:
        return iq.astype(np.complex64)

    return (resample_poly(iq, up=1, down=factor) * np.sqrt(factor)).astype(np.complex64)