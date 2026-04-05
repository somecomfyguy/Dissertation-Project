
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

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