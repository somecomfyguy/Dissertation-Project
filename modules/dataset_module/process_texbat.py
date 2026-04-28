"""
Texas Spoofing Test Battery (TEXBAT) dataset loader for cross-dataset
spoofing generalisation evaluation. It provides live-sky GPS L1 C/A 
recordings from the UT Austin Radionavigation Laboratory, not simulator output. 
Zero-shot: the pre-trained model sees TEXBAT for the first time at evaluation.

Binary file format: interleaved signed 16-bit integers (I, Q, I, Q, ...)
Native sample rate:  25 MHz complex (decimated 5x to 5 MHz on read)
Bytes per native complex sample: 4 (2 x int16)

Onset times for ds1-ds4 taken from:
    Lemmenes, Corbell & Gunawardena (2016), "Detailed Analysis of the
    TEXBAT Datasets Using a High Fidelity Software GPS Receiver,"
    Proceedings of ION GNSS+ 2016, Table 2.

ds5 and ds6 (dynamic scenarios) are not analysed in that reference; we
default to 120 s onset to match the OAKBAT convention.

Expected directory layout (matches the local disk):
    <texbat_root>/
        L1/
            cleanStatic.bin      (all clean — static antenna baseline)
            cleanDynamic.bin     (all clean — dynamic antenna baseline)
            ds1.bin  ... ds6.bin (spoofing scenarios)
"""
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import numpy as np

from modules.common.types import Segment, SAMPLE_RATE
from modules.common.decimation import polyphase_decimate


# TEXBAT-specific constants
TEXBAT_NATIVE_FS  = 25e6
DECIMATION_FACTOR = int(TEXBAT_NATIVE_FS / SAMPLE_RATE)  # 5

# Relative path to GPS L1 subdir within the dataset root.
texbat_l1_filepath = Path("L1")


@dataclass
class TexbatScenario:
    """
    Metadata for a single TEXBAT recording file.

    Attributes:
        data_path:   Relative path to the .bin file (e.g. 'L1/ds1.bin').
        scenario:    Scenario identifier (e.g. 'ds1', 'cleanStatic').
        spoof_class: Unified taxonomy label for post-onset windows.
                     'clean' for the cleanStatic / cleanDynamic files.
        onset_time:  Time in seconds from the start of the recording at
                     which spoofing begins, or None if the file is entirely
                     clean. ds1-ds4 from Lemmenes et al. (2016) Table 2;
                     ds5/ds6 default to 120 s (OAKBAT convention).
    """
    data_path:   str
    scenario:    str
    spoof_class: str
    onset_time:  Optional[float]


# TEXBAT scenarios — same taxonomy mapping as OAKBAT (which was designed
# to mirror TEXBAT ds1-ds6). ds4 and ds6 both map to 'spoof_dynamic' per
# the merged-class decision.
SCENARIOS_TEXBAT = [
    TexbatScenario(os.path.join(texbat_l1_filepath, "cleanStatic.bin"),
                   "cleanStatic",  "clean", onset_time=None),
    TexbatScenario(os.path.join(texbat_l1_filepath, "cleanDynamic.bin"),
                   "cleanDynamic", "clean", onset_time=None),
    TexbatScenario(os.path.join(texbat_l1_filepath, "ds1.bin"),
                   "ds1", "spoof_overpowered_instant", onset_time=125.0),
    TexbatScenario(os.path.join(texbat_l1_filepath, "ds2.bin"),
                   "ds2", "spoof_overpowered_gradual", onset_time=110.1),
    TexbatScenario(os.path.join(texbat_l1_filepath, "ds3.bin"),
                   "ds3", "spoof_matched_time",        onset_time=118.9),
    TexbatScenario(os.path.join(texbat_l1_filepath, "ds4.bin"),
                   "ds4", "spoof_dynamic",             onset_time=113.8),
    TexbatScenario(os.path.join(texbat_l1_filepath, "ds5.bin"),
                   "ds5", "spoof_position_push",       onset_time=120.0),
    TexbatScenario(os.path.join(texbat_l1_filepath, "ds6.bin"),
                   "ds6", "spoof_dynamic",             onset_time=120.0),
]


def read_texbat_chunk(filepath: str, start_sample: int,
                      num_samples: int) -> np.ndarray:
    """
    Read an IQ window from a TEXBAT .bin file and decimate to 5 MHz.

    Both start_sample and num_samples are specified in OUTPUT (5 MHz)
    units to keep the pipeline consistent with OAKBAT / Swinney. This
    function internally reads DECIMATION_FACTOR times as many native
    samples from disk, then applies polyphase 5x decimation before
    returning.

    Args:
        filepath:     Path to the TEXBAT .bin file.
        start_sample: Zero-based start offset in 5 MHz sample units.
        num_samples:  Number of complex samples to return at 5 MHz.

    Returns:
        Complex64 array of length num_samples at 5 MHz.
    """
    native_start = start_sample * DECIMATION_FACTOR
    native_count = num_samples  * DECIMATION_FACTOR

    offset_bytes = native_start * 4    # 2 x int16 per complex sample
    read_bytes   = native_count * 4

    with open(filepath, "rb") as f:
        f.seek(offset_bytes)
        raw = np.frombuffer(f.read(read_bytes), dtype=np.int16)

    iq = raw.reshape(-1, 2)
    native_iq = iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)

    decimated = polyphase_decimate(native_iq, DECIMATION_FACTOR)
    # Defensive slice to guarantee exact length even if filter edge rounding
    # produces one extra sample.
    return decimated[:num_samples]


def load_texbat_iq(filepath: str,
                   max_duration_s: Optional[float] = None) -> np.ndarray:
    """
    Load a TEXBAT file into a complex64 array at 5 MHz (post-decimation).

    Intended for sanity checks and plotting; the streaming pipeline uses
    read_texbat_chunk() to avoid loading 40+ GB files into RAM.

    Args:
        filepath:       Path to the .bin file.
        max_duration_s: If provided, only the first max_duration_s seconds
                        of the native signal are decoded before decimation.

    Returns:
        Complex64 array at 5 MHz.
    """
    max_native_samples = None
    if max_duration_s is not None:
        max_native_samples = int(max_duration_s * TEXBAT_NATIVE_FS)

    raw = np.fromfile(filepath, dtype=np.int16)
    if max_native_samples is not None:
        raw = raw[:max_native_samples * 2]

    raw = raw[:len(raw) - len(raw) % 2]
    iq  = raw.reshape(-1, 2)
    native_iq = iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)

    return polyphase_decimate(native_iq, DECIMATION_FACTOR)


def scan_texbat_segments(texbat_dir: str,
                         segment_length_s: float = 0.02,
                         overlap_s: float = 0.0,
                         max_file_duration_s: Optional[float] = None,
                         ) -> list[Segment]:
    """
    Scan TEXBAT files and produce Segment metadata WITHOUT loading IQ.

    File sizes are used to determine total sample counts; segment
    boundaries and labels are computed from the scenario onset times.
    The resulting Segment objects have data=None — IQ is read on demand
    (with on-the-fly 5x decimation) by read_texbat_chunk().

    Windows straddling the spoofing onset are discarded. For clean-only
    scenarios the entire file is labelled 'clean'.

    Args:
        texbat_dir:          Root directory containing the L1/ subdir.
        segment_length_s:    Window duration in seconds (default 20 ms).
        overlap_s:           Fractional overlap in [0.0, 1.0).
        max_file_duration_s: Cap on how much of each file to use. Applied
                             in the OUTPUT (5 MHz) timeline.

    Returns:
        List of Segment objects with data=None and dataset='texbat'.
        start_sample and num_samples are in 5 MHz units.
    """
    texbat_path    = Path(texbat_dir)
    window_samples = int(segment_length_s * SAMPLE_RATE)   # 100k @ 5 MHz
    hop_samples    = int(window_samples * (1 - overlap_s))

    all_meta: list[Segment] = []

    for scenario in SCENARIOS_TEXBAT:
        filepath = texbat_path / scenario.data_path
        if not filepath.exists():
            print(f"  [SKIP] {filepath} not found")
            continue

        # Determine total samples from file size:
        # native samples = bytes / 4 (2 x int16 per complex);
        # output samples = native / DECIMATION_FACTOR.
        file_size_bytes      = filepath.stat().st_size
        native_total_samples = file_size_bytes // 4
        output_total_samples = native_total_samples // DECIMATION_FACTOR

        if max_file_duration_s is not None:
            max_samples          = int(max_file_duration_s * SAMPLE_RATE)
            output_total_samples = min(output_total_samples, max_samples)

        # For clean-only files, push onset beyond any reachable sample so
        # every window is labelled 'clean' by the logic below.
        if scenario.onset_time is None:
            onset_sample = np.iinfo(np.int64).max
        else:
            onset_sample = int(scenario.onset_time * SAMPLE_RATE)

        n_clean   = 0
        n_spoofed = 0
        start     = 0

        while start + window_samples <= output_total_samples:
            end = start + window_samples

            if end <= onset_sample:
                label      = "clean"
                is_spoofed = False
                n_clean   += 1
            elif start >= onset_sample:
                label      = scenario.spoof_class
                is_spoofed = True
                n_spoofed += 1
            else:
                # Straddles the onset — drop to avoid ambiguous labels.
                start += hop_samples
                continue

            all_meta.append(Segment(
                data=None,
                label=label,
                source_file=str(filepath),
                scenario=scenario.scenario,
                start_sample=start,
                num_samples=window_samples,
                is_spoofed=is_spoofed,
                dataset="texbat",
            ))
            start += hop_samples

        print(f"  Scanned {filepath.name} ({scenario.scenario}): "
              f"{n_clean + n_spoofed} segments "
              f"(clean: {n_clean}, spoofed: {n_spoofed})")

    print(f"  Total TEXBAT segments: {len(all_meta)}")
    return all_meta