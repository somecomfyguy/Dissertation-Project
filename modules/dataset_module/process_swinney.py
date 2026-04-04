"""
Procesing for Swinney dataset
"""

from scipy.io import loadmat
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
from typing import Optional


SWINNEY_CLASS_MAP = {
    "NoJam":       "clean",
    "SingleAM":    "jam_am",
    "SingleChirp": "jam_chirp",
    "SingleFM":    "jam_fm",
    "DME":         "jam_dme",
    "NB":          "jam_narrowband",
}


# ---------------------------------------------------------------------------
# Segment dataclass
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Swinney loader
# ---------------------------------------------------------------------------
def load_swinney_segments(swinney_dir: str,
                          split: str = "training") -> list[Segment]:
    """
    Load all .mat files from a Swinney split directory into Segment objects.
    The `features` field is left as None; call compute_features() separately
    if needed (see pipeline notes in module docstring).
    """
    class_map = SWINNEY_CLASS_MAP
    base_dir  = Path(swinney_dir) / split.capitalize()

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Swinney {split} directory not found: {base_dir}\n"
            f"Expected: {swinney_dir}/{split.capitalize()}/<class>/*.mat"
        )

    segments     = []
    mat_var_name = "GNSS_plus_Jammer_awgn"

    for class_dir_name, unified_label in class_map.items():
        class_dir = base_dir / class_dir_name
        if not class_dir.exists():
            print(f"  [WARN] Class directory not found, skipping: {class_dir}")
            continue

        mat_files = sorted(class_dir.glob("*.mat"))
        if not mat_files:
            print(f"  [WARN] No .mat files in {class_dir}")
            continue

        loaded = 0
        for mat_path in mat_files:
            try:
                mat_data = loadmat(str(mat_path))
            except Exception as e:
                print(f"  [WARN] Failed to load {mat_path.name}: {e}")
                continue

            if mat_var_name not in mat_data:
                data_keys = [k for k in mat_data if not k.startswith("__")]
                if len(data_keys) == 1:
                    iq = mat_data[data_keys[0]].squeeze()
                else:
                    print(f"  [WARN] '{mat_var_name}' not found in "
                          f"{mat_path.name}. Keys: {data_keys}")
                    continue
            else:
                iq = mat_data[mat_var_name].squeeze()

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
    """Read a single Swinney .mat file and return complex IQ."""
    mat_data = loadmat(filepath)
    mat_var_name = "GNSS_plus_Jammer_awgn"

    if mat_var_name not in mat_data:
        data_keys = [k for k in mat_data.keys() if not k.startswith("__")]
        if len(data_keys) == 1:
            iq = mat_data[data_keys[0]].squeeze()
        else:
            raise KeyError(f"Variable '{mat_var_name}' not found. Keys: {data_keys}")
    else:
        iq = mat_data[mat_var_name].squeeze()

    if not np.iscomplexobj(iq):
        if iq.ndim == 2 and iq.shape[1] == 2:
            iq = iq[:, 0] + 1j * iq[:, 1]
        else:
            iq = iq.astype(np.complex64)
    return iq.astype(np.complex64)