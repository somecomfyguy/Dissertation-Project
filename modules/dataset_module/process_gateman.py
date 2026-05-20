"""
GATEMAN dataset loader for cross-dataset jamming generalisation evaluation.

The GATEMAN Raw I/Q dataset (Tampere University, 2019, Zenodo DOI:
10.5281/zenodo.2654322 v2) provides lab-recorded GPS L1, Galileo E1
jamming scenarios at 20 MS/s (decimated 4x to 5 MHz on read).

Each scenario folder contains TWO files:
    - A clean GNSS baseline (simulator-generated single PRN)
    - A jammer signal (AM tone, chirp 10 MHz, or chirp 20 MHz)

These are recorded SEPARATELY and must be mixed at a user-controlled
jammer-to-signal ratio (JSR) to produce a jammed signal. This enables
JSR sweep experiments that test detection thresholds.

Dataset references:
    Borio et al. (2019), "GATEMAN project -- Wide-bandwidth, high-precision
    GNSS and jammer raw data", Zenodo. DOI: 10.5281/zenodo.2654322
"""
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import numpy as np

from modules.common.types import Segment, SAMPLE_RATE
from modules.common.decimation import polyphase_decimate


# GATEMAN-specific constants
GATEMAN_NATIVE_FS  = 20e6
DECIMATION_FACTOR  = int(GATEMAN_NATIVE_FS / SAMPLE_RATE)  # 4


@dataclass
class GatemanScenario:
    """
    Metadata for a single GATEMAN scenario folder.

    Attributes:
        folder:       Subfolder name (e.g. 'GPSL1+AMtone').
        gnss_file:    Filename of the clean GNSS baseline.
        jammer_file:  Filename of the jammer signal.
        jam_class:    Unified taxonomy label (e.g. 'jam_am', 'jam_chirp').
        constellation: 'gps' or 'galileo'.
    """
    folder:        str
    gnss_file:     str
    jammer_file:   str
    jam_class:     str
    constellation: str


# Both Chirp10MHz and Chirp20MHz map to 'jam_chirp' in the unified taxonomy.
# The model was trained on Swinney's chirp which has variable sweep bandwidth,
# so both GATEMAN chirp variants test the same class boundary.
SCENARIOS_GATEMAN = [
    GatemanScenario("GPSL1+AMtone",     "GPSL1@20MSps-16bit.bin",
                    "AMtone@20MSps-16bit.bin",     "jam_am",    "gps"),
    GatemanScenario("GPSL1+Chirp10MHz", "GPSL1@20MSps-16bit.bin",
                    "Chirp10MHz@20MSps-16bit.bin", "jam_chirp", "gps"),
    GatemanScenario("GPSL1+Chirp20MHz", "GPSL1@20MSps-16bit.bin",
                    "Chirp20MHz@20MSps-16bit.bin", "jam_chirp", "gps"),
    GatemanScenario("GALE1+AMtone",     "GALE1@20MSps-16bit.bin",
                    "AMtone@20MSps-16bit.bin",     "jam_am",    "galileo"),
    GatemanScenario("GALE1+Chirp10MHz", "GALE1@20MSps-16bit.bin",
                    "Chirp10MHz@20MSps-16bit.bin", "jam_chirp", "galileo"),
    GatemanScenario("GALE1+Chirp20MHz", "GALE1@20MSps-16bit.bin",
                    "Chirp20MHz@20MSps-16bit.bin", "jam_chirp", "galileo"),
]


def _read_gateman_raw(filepath: str, start_sample: int,
                      num_samples: int) -> np.ndarray:
    """
    Read raw IQ from a GATEMAN .bin file at native 20 MS/s.
    GATEMAN uses BIG-ENDIAN int16.

    Args:
        filepath:     Path to the .bin file.
        start_sample: Zero-based start in native (20 MHz) sample units.
        num_samples:  Number of native complex samples to read.

    Returns:
        Complex64 array of shape (num_samples,) at native rate.
    """
    offset_bytes = start_sample * 4
    read_bytes   = num_samples * 4

    with open(filepath, "rb") as f:
        f.seek(offset_bytes)
        raw = np.frombuffer(f.read(read_bytes), dtype=np.dtype(">i2"))

    iq = raw.reshape(-1, 2)
    return iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)


def mix_at_jsr(gnss_iq: np.ndarray, jammer_iq: np.ndarray,
               jsr_db: float) -> np.ndarray:
    """
    Combine GNSS and jammer IQ signals at a specified jammer-to-signal ratio.

    The JSR is defined as 10·log₁₀(P_jammer / P_gnss). The jammer signal
    is scaled to achieve the target JSR relative to the GNSS signal's
    measured power, then added sample-by-sample.

    Args:
        gnss_iq:   Complex64 clean GNSS signal.
        jammer_iq: Complex64 jammer signal (same length).
        jsr_db:    Target jammer-to-signal ratio in dB.

    Returns:
        Complex64 combined signal at the target JSR.
    """
    gnss_power   = np.mean(np.abs(gnss_iq) ** 2)
    jammer_power = np.mean(np.abs(jammer_iq) ** 2)

    if jammer_power < 1e-12 or gnss_power < 1e-12:
        return gnss_iq  # avoid division by zero on silent signals

    # Target jammer power = gnss_power * 10^(JSR/10)
    target_jammer_power = gnss_power * (10.0 ** (jsr_db / 10.0))
    scale = np.sqrt(target_jammer_power / jammer_power)

    return (gnss_iq + scale * jammer_iq).astype(np.complex64)


def read_gateman_chunk(gnss_path: str, jammer_path: str,
                       start_sample: int, num_samples: int,
                       jsr_db: float = 20.0) -> np.ndarray:
    """
    Read a window from GATEMAN, calibrate GNSS amplitude to match
    OAKBAT regime, mix jammer at JSR on top, then decimate to 5 MHz.

    Calibration order matters: the GNSS baseline is scaled to match
    OAKBAT's amplitude regime BEFORE mixing, so that the jammer's
    power is added at the correct absolute level relative to a
    training-compatible GNSS signal. Calibrating after mixing would
    compress or expand both components together, distorting the
    spectral shape the classifier was trained on.
    """
    native_start = start_sample * DECIMATION_FACTOR
    native_count = num_samples  * DECIMATION_FACTOR

    gnss_raw   = _read_gateman_raw(gnss_path,   native_start, native_count)
    jammer_raw = _read_gateman_raw(jammer_path,  native_start, native_count)

    # Calibrate GNSS to OAKBAT amplitude regime BEFORE mixing
    gnss_rms = np.sqrt(np.mean(np.abs(gnss_raw) ** 2))
    if gnss_rms > 1e-6:
        target_rms = 2300.0
        gnss_raw *= (target_rms / gnss_rms)

    # Scale jammer to match the same calibration ratio so JSR
    # computation in mix_at_jsr operates on consistent amplitudes
    jammer_rms = np.sqrt(np.mean(np.abs(jammer_raw) ** 2))
    if jammer_rms > 1e-6:
        jammer_raw *= (target_rms / jammer_rms)

    # Now mix at the requested JSR — both signals are in the same
    # amplitude regime, so the JSR is physically meaningful
    mixed = mix_at_jsr(gnss_raw, jammer_raw, jsr_db)

    decimated = polyphase_decimate(mixed, DECIMATION_FACTOR)
    return decimated[:num_samples]


def read_gateman_clean_chunk(gnss_path: str,
                             start_sample: int,
                             num_samples: int) -> np.ndarray:
    """
    Read a clean GNSS-only window from GATEMAN, calibrate amplitude,
    then decimate to 5 MHz.
    """
    native_start = start_sample * DECIMATION_FACTOR
    native_count = num_samples  * DECIMATION_FACTOR

    gnss_raw = _read_gateman_raw(gnss_path, native_start, native_count)

    # Same amplitude calibration as jammed path
    rms = np.sqrt(np.mean(np.abs(gnss_raw) ** 2))
    if rms > 1e-6:
        target_rms = 2300.0
        gnss_raw *= (target_rms / rms)

    decimated = polyphase_decimate(gnss_raw, DECIMATION_FACTOR)
    return decimated[:num_samples]


def scan_gateman_segments(gateman_dir: str,
                          jsr_db: float = 20.0,
                          constellations: list[str] = ["gps", "galileo"],
                          segment_length_s: float = 0.02,
                          overlap_s: float = 0.0,
                          max_file_duration_s: Optional[float] = None,
                          include_clean: bool = True,
                          ) -> list[Segment]:
    """
    Scan GATEMAN scenario folders and produce Segment metadata.

    For each scenario, produces:
        - Jammed segments (GNSS + jammer mixed at jsr_db)
        - Clean segments (GNSS-only, if include_clean=True)

    Segment.source_file stores the GNSS path. Segment.scenario stores
    the jammer path — overloaded for use by read_segment_iq() dispatch.
    The JSR is embedded in the scenario string for traceability.

    Args:
        gateman_dir:         Root directory containing the 6 scenario folders.
        jsr_db:              Jammer-to-signal ratio for mixing.
        constellations:      Which constellations to include.
        segment_length_s:    Window duration (default 20 ms).
        overlap_s:           Fractional overlap in [0.0, 1.0).
        max_file_duration_s: Cap on file duration in output (5 MHz) units.
        include_clean:       If True, also produce clean segments from the
                             GNSS baseline files.

    Returns:
        List of Segment objects with data=None and dataset='gateman'.
        start_sample and num_samples are in 5 MHz units.
    """
    gateman_path   = Path(gateman_dir)
    window_samples = int(segment_length_s * SAMPLE_RATE)
    hop_samples    = int(window_samples * (1 - overlap_s))

    scenarios = [s for s in SCENARIOS_GATEMAN
                 if s.constellation in constellations]

    all_meta: list[Segment] = []
    clean_files_done: set[str] = set()  # avoid duplicate clean from same file

    for scenario in scenarios:
        folder_path  = gateman_path / scenario.folder
        gnss_path    = folder_path / scenario.gnss_file
        jammer_path  = folder_path / scenario.jammer_file

        if not gnss_path.exists() or not jammer_path.exists():
            print(f"  [SKIP] {scenario.folder}: files not found")
            continue

        # Determine total output samples from file size
        file_size_bytes      = gnss_path.stat().st_size
        native_total_samples = file_size_bytes // 4
        output_total_samples = native_total_samples // DECIMATION_FACTOR

        if max_file_duration_s is not None:
            max_samples          = int(max_file_duration_s * SAMPLE_RATE)
            output_total_samples = min(output_total_samples, max_samples)

        # Jammed segments
        n_jammed = 0
        start    = 0
        while start + window_samples <= output_total_samples:
            all_meta.append(Segment(
                data=None,
                label=scenario.jam_class,
                source_file=str(gnss_path),
                scenario=f"{scenario.folder}|{jammer_path}|jsr={jsr_db}",
                start_sample=start,
                num_samples=window_samples,
                is_spoofed=False,
                dataset="gateman",
            ))
            n_jammed += 1
            start    += hop_samples

        print(f"  Scanned {scenario.folder}: {n_jammed} jammed segments "
              f"({scenario.jam_class}, JSR={jsr_db} dB)")

        # Clean segments (once per unique GNSS file)
        if include_clean and str(gnss_path) not in clean_files_done:
            clean_files_done.add(str(gnss_path))
            n_clean = 0
            start   = 0
            while start + window_samples <= output_total_samples:
                all_meta.append(Segment(
                    data=None,
                    label="clean",
                    source_file=str(gnss_path),
                    scenario=f"{scenario.folder}|clean|jsr=none",
                    start_sample=start,
                    num_samples=window_samples,
                    is_spoofed=False,
                    dataset="gateman_clean",
                ))
                n_clean += 1
                start   += hop_samples
            print(f"  Scanned {scenario.gnss_file} (clean): {n_clean} segments")

    print(f"  Total GATEMAN segments: {len(all_meta)}")
    return all_meta