"""
Procesing for Oakbat dataset
"""

import json
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import numpy as np

# Global variables
SAMPLE_RATE = 5e6  # 5 MHz
oakbat_gps_filepath = Path("L1")
oakbat_gal_filepath = Path("E1")

# SCENARIO METADATA
@dataclass
class OakbatScenario:
    data_path: str
    scenario: str
    spoof_class: str
    onset_time: float


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

ALL_SCENARIOS = SCENARIOS_GPS + SCENARIOS_GALILEO


# IQ LOADER
def load_oakbat_iq(filepath: str, fs: float = SAMPLE_RATE,
                   max_duration_s: Optional[float] = None) -> np.ndarray:
    """
    Load OAKBAT raw binary IQ file.
    Format: interleaved 16-bit signed integers (I, Q, I, Q, ...)
    Returns: complex64 numpy array.
    """
    max_samples = None
    if max_duration_s is not None:
        max_samples = int(max_duration_s * fs)

    raw = np.fromfile(filepath, dtype=np.int16)
    if max_samples is not None:
        raw = raw[:max_samples * 2]

    raw = raw[:len(raw) - len(raw) % 2]
    iq = raw.reshape(-1, 2)
    return (iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32))


# SEGMENT DATACLASS
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

def segment_signal(signal: np.ndarray, scenario: OakbatScenario,
                   fs: float = SAMPLE_RATE, segment_length_s: float = 0.02,
                   overlap_s: float = 0.0,
                   clean_label: str = "clean") -> list[Segment]:
    """
    Segment a recording into fixed-length windows, labelling each window
    as clean (pre-onset) or the scenario spoofing class (post-onset).
    Windows that straddle the onset boundary are discarded.
    """
    window_samples = int(segment_length_s * fs)
    hop_samples    = int(window_samples * (1 - overlap_s))
    onset_sample   = int(scenario.onset_time * fs)

    segments = []
    start = 0

    while start + window_samples <= len(signal):
        end   = start + window_samples
        chunk = signal[start:end]

        if end <= onset_sample:
            label = clean_label
            is_spoofed = False
        elif start >= onset_sample:
            label = scenario.spoof_class
            is_spoofed = True
        else:
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
    Read a specific chunk of IQ from an OAKBAT .bin file without
    loading the entire file. Seeks directly to the offset.
    """
    # Each complex sample = 2 × int16 = 4 bytes
    offset_bytes = start_sample * 2 * 2  # 2 components × 2 bytes each
    count = num_samples * 2              # I and Q interleaved

    with open(filepath, "rb") as f:
        f.seek(offset_bytes)
        raw = np.frombuffer(f.read(count * 2), dtype=np.int16)

    iq = raw.reshape(-1, 2)
    return iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)