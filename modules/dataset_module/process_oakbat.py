"""
Oak Ridge Spoofing and Interference Test Battery (OAKBAT) dataset loader and segmentation utilities.

Binary file format: interleaved signed 16-bit integers (I, Q, I, Q, …)
Each complex sample therefore occupies 4 bytes (2 x int16).

Dataset DOIs:
    GPS L1:     10.13139/ORNLNCCS/1664429
    Galileo E1: 10.13139/ORNLNCCS/1665888

Expected directory layout:
    <oakbat_root>/
        L1/
            os1.bin  … os6.bin     (GPS L1 scenarios ds1-ds6)
        E1/
            os9a.bin, os10.bin … os14.bin  (Galileo E1 scenarios ds1-ds6)
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import numpy as np

# Global variables
SAMPLE_RATE = 5e6  # 5 MHz sampling frequency for all files

# Relative paths to GPS and Galileo subdirectories within the dataset root.
# These are used to build per-scenario file paths in SCENARIOS_GPS/GALILEO.
oakbat_gps_filepath = Path("L1")
oakbat_gal_filepath = Path("E1")

# SCENARIO METADATA
@dataclass
class OakbatScenario:
    """
    Metadata for a single OAKBAT recording file.

    Each scenario corresponds to one .bin file and one spoofing attack type.
    The onset_time marks the boundary between the clean pre-spoofing segment
    and the post-onset spoofed segment within the recording.

    Attributes:
        data_path:   Relative path to the .bin file (e.g. 'L1/os1.bin').
        scenario:    Scenario identifier string (e.g. 'ds1').
        spoof_class: Unified taxonomy label for the spoofing type (e.g.
                     'spoof_overpowered_instant'). Used as the class label
                     for all post-onset windows from this file.
        onset_time:  Time in seconds from the start of the recording at
                     which spoofing begins. Windows that straddle this
                     boundary are discarded during segmentation.
    """
    data_path: str
    scenario: str
    spoof_class: str
    onset_time: float

# OAKBAT GPS L1 scenarios (os1.bin – os6.bin under L1/)
# Scenario descriptions mirror Texas Spoofing Test Battery (TEXBAT):
#   ds1: Instantaneous overpowered spoofing
#   ds2: Gradual overpowered spoofing (+10 dB)
#   ds3: Power-matched spoofing (time push, +1.3 dB)
#   ds4: Power-matched spoofing (dynamic)
#   ds5: Position push
#   ds6: Dynamic position push
SCENARIOS_GPS = [
    OakbatScenario(os.path.join(oakbat_gps_filepath, "os1.bin"),  "ds1",
                   "spoof_overpowered_instant",  onset_time=120.0),
    OakbatScenario(os.path.join(oakbat_gps_filepath, "os2.bin"),  "ds2",
                   "spoof_overpowered_gradual",  onset_time=120.0),
    OakbatScenario(os.path.join(oakbat_gps_filepath, "os3.bin"),  "ds3",
                   "spoof_matched_time",         onset_time=120.0),
    OakbatScenario(os.path.join(oakbat_gps_filepath, "os4.bin"),  "ds4",
                   "spoof_matched_dynamic",      onset_time=120.0),
    OakbatScenario(os.path.join(oakbat_gps_filepath, "os5.bin"),  "ds5",
                   "spoof_position_push",        onset_time=120.0),
    OakbatScenario(os.path.join(oakbat_gps_filepath, "os6.bin"),  "ds6",
                   "spoof_dynamic_position",     onset_time=120.0),
]

# OAKBAT Galileo E1 scenarios (os9a.bin, os10.bin – os14.bin under E1/)
# Mirrors GPS taxonomy; os9a is the first Galileo file (no os7/os8).
SCENARIOS_GALILEO = [
    OakbatScenario(os.path.join(oakbat_gal_filepath, "os9a.bin"), "ds1",
                   "spoof_overpowered_instant",  onset_time=120.0),
    OakbatScenario(os.path.join(oakbat_gal_filepath, "os10.bin"), "ds2",
                   "spoof_overpowered_gradual",  onset_time=120.0),
    OakbatScenario(os.path.join(oakbat_gal_filepath, "os11.bin"), "ds3",
                   "spoof_matched_time",         onset_time=120.0),
    OakbatScenario(os.path.join(oakbat_gal_filepath, "os12.bin"), "ds4",
                   "spoof_matched_dynamic",      onset_time=120.0),
    OakbatScenario(os.path.join(oakbat_gal_filepath, "os13.bin"), "ds5",
                   "spoof_position_push",        onset_time=120.0),
    OakbatScenario(os.path.join(oakbat_gal_filepath, "os14.bin"), "ds6",
                   "spoof_dynamic_position",     onset_time=120.0),
]

# Combined list used by scan/pipeline functions when iterating all scenarios.
ALL_SCENARIOS = SCENARIOS_GPS + SCENARIOS_GALILEO


# SEGMENT DATACLASS
@dataclass
class Segment:
    """
    A fixed-length IQ window extracted from an OAKBAT recording.

    Attributes:
        data:         Complex IQ samples as a complex64 
        label:        Class label (e.g. 'clean' or 'spoof_overpowered_instant').
        source_file:  Path to the originating .bin file.
        scenario:     Scenario identifier of the parent recording (e.g. 'ds1').
        start_sample: Sample offset within the source file at which this
                      window begins, for tracing segments back to the original recording.
        is_spoofed:   True if the window lies entirely after the spoofing
                      onset; False for clean pre-onset windows.
        features:     Optional 8-element float32 feature vector produced by
                      compute_features(). None until explicitly computed so
                      existing code that does not use features is unaffected.
    """
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


def segment_signal(signal: np.ndarray, scenario: OakbatScenario,
                   fs: float = SAMPLE_RATE, segment_length_s: float = 0.02,
                   overlap_s: float = 0.0,
                   clean_label: str = "clean") -> list[Segment]:
    """
    Slice a full OAKBAT recording into fixed-length labelled windows.\n
    Each window is labelled according to its position relative to the
    spoofing onset time:
    - Entirely before onset => label = clean_label, is_spoofed = False
    - Entirely after onset => label = scenario.spoof_class, is_spoofed = True
    - Straddles the boundary => discarded to avoid ambiguous labels

    Args:
        signal: Complex IQ array of the full recording.
        scenario: OakbatScenario providing onset_time and spoof_class.
        fs: Sampling frequency in Hz.
        segment_length_s: Window duration in seconds (default: 20 ms).
        overlap_s: Fractional overlap between consecutive windows, 
            in [0.0, 1.0). 0.0 means no overlap (hop = window).
        clean_label: Label string assigned to pre-onset windows.

    Returns:
        List of Segment objects with IQ data and labels populated.
        The features field is left as None; call compute_features() separately.
    """
    window_samples = int(segment_length_s * fs)
    hop_samples = int(window_samples * (1 - overlap_s))
    onset_sample = int(scenario.onset_time * fs)

    segments: list[Segment] = []
    start = 0

    while start + window_samples <= len(signal):
        end   = start + window_samples
        chunk = signal[start:end]

        if end <= onset_sample:
            label      = clean_label
            is_spoofed = False
        elif start >= onset_sample:
            label      = scenario.spoof_class
            is_spoofed = True
        else:
            # Window straddles the onset boundary, will be skipped
            start += hop_samples
            continue

        segments.append(Segment(
            data=chunk,
            label=label,
            source_file=scenario.data_path,
            scenario=scenario.scenario,
            start_sample=start,
            is_spoofed=is_spoofed,
        ))
        start += hop_samples

    return segments

def read_oakbat_chunk(filepath: str, start_sample: int,
                      num_samples: int) -> np.ndarray:
    """
    Read a single IQ window from an OAKBAT .bin file via direct byte-offset
    seeking, without loading the full file into memory.

    This is the streaming read used in pass 2 of the pipeline: given a
    SegmentMeta produced during scanning, it fetches only the bytes needed
    for that window.

    Args:
        filepath: Path to the .bin file.
        start_sample: Zero-based index of the first complex sample to read.
        num_samples: Number of complex samples to read.

    Returns:
        Complex64 numpy array of shape (num_samples,).
    """
    # Each complex sample = 2 × int16 = 4 bytes total.
    offset_bytes = start_sample * 4     # 2 components × 2 bytes each
    count        = num_samples * 2      # number of int16 values to read

    with open(filepath, "rb") as f:
        f.seek(offset_bytes)
        raw = np.frombuffer(f.read(count * 2), dtype=np.int16)

    iq = raw.reshape(-1, 2)
    return iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)


# IQ LOADER
def load_oakbat_iq(filepath: str, fs: float = SAMPLE_RATE,
                   max_duration_s: Optional[float] = None) -> np.ndarray:
    """
    Load an OAKBAT raw binary IQ file into a complex64 array.
    The binary format stores samples as interleaved signed 16-bit integers:
        [I_0, Q_0, I_1, Q_1, …]

    Args:
        filepath: Path to the .bin file.
        fs: Sampling frequency in Hz. Used only to compute the
            sample count from max_duration_s; does not affect
            the decoded values.
        max_duration_s: If provided, only the first max_duration_s seconds
                        of the file are loaded.

    Returns:
        Complex64 numpy array of shape (N,) where N ≤ file_samples.
    """
    max_samples = None
    if max_duration_s is not None:
        max_samples = int(max_duration_s * fs)

    raw = np.fromfile(filepath, dtype=np.int16)
    if max_samples is not None:
        raw = raw[:max_samples * 2]   # 2 int16 values per complex sample

    # Drop any trailing odd int16 to guarantee even length before reshape.
    raw = raw[:len(raw) - len(raw) % 2]
    iq = raw.reshape(-1, 2)
    return iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)
