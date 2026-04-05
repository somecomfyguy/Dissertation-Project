"""
Swinney dataset loader for GNSS jamming classification.

The Swinney dataset provides GPS L1 IQ recordings of six jamming types
plus a clean (no-jam) baseline, stored as MATLAB .mat files. Each file
contains a single variable ('GNSS_plus_Jammer_awgn') with a complex IQ
array of a complete simulated scenario.

Dataset references:
    Swinney & Woods (2021), "Raw IQ dataset for GNSS GPS jamming signal
    classification", Zenodo. DOI: 10.5281/zenodo.4629685

    Swinney & Woods (2021), "GNSS Jamming Classification via CNN, Transfer
    Learning & the Novel Concatenation of Signal Representations", CyberSA.
    DOI: 10.1109/CyberSA52016.2021.9478250

Expected directory layout:
    <swinney_root>/
        Training/
            NoJam/       Training_raw_1.mat … Training_raw_1000.mat
            SingleAM/    …
            SingleChirp/ …
            SingleFM/    …
            DME/         …
            NB/          …
        Testing/
            NoJam/       Testing_raw_1.mat … Testing_raw_250.mat
            …
"""

from scipy.io import loadmat
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from common.types import *


# Maps Swinney directory names to the unified project class labels.
# The right-hand values must match SWINNEY_CLASS_MAP in dataset_main.py.
SWINNEY_CLASS_MAP: dict[str, str] = {
    "NoJam":       "clean",
    "SingleAM":    "jam_am",
    "SingleChirp": "jam_chirp",
    "SingleFM":    "jam_fm",
    "DME":         "jam_dme",
    "NB":          "jam_narrowband",
}


def load_swinney_segments(swinney_dir: str,
                          split: str = "training") -> list[Segment]:
    """
    Load all .mat files from a Swinney split directory into Segment objects.

    Iterates over all class subdirectories listed in SWINNEY_CLASS_MAP,
    reads each .mat file, and converts the stored IQ array to complex64.
    The expected MATLAB variable name is 'GNSS_plus_Jammer_awgn'; if absent,
    the function falls back to the only non-metadata key in the file, or
    raises a KeyError if multiple ambiguous keys are present.

    The features field is left as None for all returned segments. Call
    compute_features() and populate the field separately if needed.

    Args:
        swinney_dir: Root directory of the Swinney dataset.
        split:       Which subset to load — 'training' or 'testing'.
                     The directory name is capitalised internally
                     (e.g. 'training' → 'Training/').

    Returns:
        List of Segment objects, one per successfully loaded .mat file.
        Files that cannot be parsed are skipped with a warning.

    Raises:
        FileNotFoundError: If the requested split directory does not exist.
    """
    mat_var_name = "GNSS_plus_Jammer_awgn"
    base_dir     = Path(swinney_dir) / split.capitalize()

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Swinney {split} directory not found: {base_dir}")

    segments: list[Segment] = []

    for class_dir_name, unified_label in SWINNEY_CLASS_MAP.items():
        class_dir = base_dir / class_dir_name
        if not class_dir.exists():
            print(f"  [WARN] Directory not found, skipping: {class_dir}")
            continue

        mat_files = sorted(class_dir.glob("*.mat"))
        if not mat_files:
            print(f"  [WARN] No .mat files found in: {class_dir}")
            continue

        loaded = 0
        for mat_path in mat_files:
            mat_data = loadmat(str(mat_path))

            # Resolve the IQ variable — use the known name or fall back.
            if mat_var_name not in mat_data:
                data_keys = [k for k in mat_data if not k.startswith("__")]
                if len(data_keys) == 1:
                    iq = mat_data[data_keys[0]].squeeze()
                else:
                    print(f"  [WARN] Unexpected keys in {mat_path.name}, skipping. "
                          f"Keys: {data_keys}")
                    continue
            else:
                iq = mat_data[mat_var_name].squeeze()

            # Ensure complex dtype — some exports store as (N, 2) real array.
            if not np.iscomplexobj(iq):
                if iq.ndim == 2 and iq.shape[1] == 2:
                    iq = iq[:, 0] + 1j * iq[:, 1]
                else:
                    iq = iq.astype(np.complex64)

            segments.append(Segment(
                data=iq.astype(np.complex64),
                label=unified_label,
                source_file=mat_path.name,
                scenario=f"swinney_{split}",
                start_sample=0,
                is_spoofed=False,
            ))
            loaded += 1

        print(f"  {unified_label} ({class_dir_name}): {loaded} files loaded")

    return segments


def read_swinney_file(filepath: str) -> np.ndarray:
    """
    Read a single Swinney .mat file and return its IQ data as a complex64 array.\n
    The expected MATLAB variable name is 'GNSS_plus_Jammer_awgn'.\n
    If absent, the function falls back to the sole non-metadata key in the file, \n
    or raises a KeyError if multiple ambiguous keys are present.

    Args:
        filepath: Path to a single Swinney .mat file.

    Returns:
        Complex64 numpy array of shape (N,) containing the IQ samples.

    Raises:
        KeyError: If the expected variable is absent and multiple candidate
                  keys exist in the .mat file.
    """
    mat_var_name = "GNSS_plus_Jammer_awgn"
    mat_data     = loadmat(filepath)

    if mat_var_name not in mat_data:
        data_keys = [k for k in mat_data if not k.startswith("__")]
        if len(data_keys) == 1:
            iq = mat_data[data_keys[0]].squeeze()
        else:
            raise KeyError(
                f"Variable '{mat_var_name}' not found in {filepath}. "
                f"Keys: {data_keys}")
    else:
        iq = mat_data[mat_var_name].squeeze()

    if not np.iscomplexobj(iq):
        if iq.ndim == 2 and iq.shape[1] == 2:
            iq = iq[:, 0] + 1j * iq[:, 1]
        else:
            iq = iq.astype(np.complex64)

    return iq.astype(np.complex64)