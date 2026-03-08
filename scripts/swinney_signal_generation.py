"""
Python translation of Swinney / Morales Ferre GNSS jamming signal generation.

Original MATLAB: signal_generation.m (Swinney & Woods, 2021)
Based on the framework from Morales Ferre, De La Fuente & Lohan (2019)

Generates synthetic GPS L1 C/A + jammer IQ samples for 6 classes:
    0 - NoJam (clean GNSS + AWGN)
    1 - SingleAM (amplitude-modulated jammer)
    2 - SingleChirp (linear chirp jammer)
    3 - SingleFM (frequency-modulated jammer)
    4 - DME/Pulse (distance measuring equipment / pulsed jammer)
    5 - NB (narrowband / continuous wave jammer)

Parameters drawn from the MATLAB code:
    - CNR:  uniform random in [25, 50] dB-Hz
    - JSR:  uniform random in [40, 80] dB
    - Fs:   sampling frequency (default matches Morales Ferre framework)
    - Nc:   number of samples per segment

NOTE: The original MATLAB code calls into ~12 sub-functions across multiple
directories (Channel/, GNSS_signals/, Jammer_signals/, etc.) that are part
of the Morales Ferre Zenodo package (DOI: 10.5281/zenodo.3783969). This
Python script recreates the *signal-level behavior* of those functions based
on the published paper descriptions and standard GNSS/jammer models. It is
not a byte-identical port — the underlying random processes differ — but it
produces statistically equivalent IQ data for ML training purposes.
"""

import numpy as np
import os
import argparse
from pathlib import Path
from scipy.io import savemat  # optional: save as .mat for compatibility


# ---------------------------------------------------------------------------
# GNSS parameters (GPS L1 C/A)
# ---------------------------------------------------------------------------
F_CARRIER = 1575.42e6      # GPS L1 carrier frequency (Hz)
F_CODE    = 1.023e6        # C/A code chip rate (Hz)
FS        = 5e6            # Sampling frequency (Hz) — matches Swinney
NC        = 50000          # Samples per segment (10 ms at 5 MHz)
NUM_SVS   = 6              # Number of visible satellites to simulate


# ---------------------------------------------------------------------------
# GPS C/A code generation (Gold code, simplified)
# ---------------------------------------------------------------------------
# PRN tap assignments for G2 delay (first 32 SVs)
PRN_TAPS = {
    1: (2,6), 2: (3,7), 3: (4,8), 4: (5,9), 5: (1,9), 6: (2,10),
    7: (1,8), 8: (2,9), 9: (3,10), 10: (2,3), 11: (3,4), 12: (5,6),
    13: (6,7), 14: (7,8), 15: (8,9), 16: (9,10), 17: (1,4), 18: (2,5),
    19: (3,6), 20: (4,7), 21: (5,8), 22: (6,9), 23: (1,3), 24: (4,6),
    25: (5,7), 26: (6,8), 27: (7,9), 28: (8,10), 29: (1,6), 30: (2,7),
    31: (3,8), 32: (4,9),
}

def generate_ca_code(prn: int) -> np.ndarray:
    """Generate 1023-chip GPS C/A Gold code for a given PRN."""
    g1 = np.ones(10, dtype=int)
    g2 = np.ones(10, dtype=int)
    tap1, tap2 = PRN_TAPS[prn]
    code = np.zeros(1023, dtype=int)
    for i in range(1023):
        g2_out = g2[tap1 - 1] ^ g2[tap2 - 1]
        code[i] = g1[9] ^ g2_out
        # Feedback
        fb1 = g1[2] ^ g1[9]
        fb2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1 = np.roll(g1, 1); g1[0] = fb1
        g2 = np.roll(g2, 1); g2[0] = fb2
    return 2 * code - 1  # Convert {0,1} to {-1,+1}


def generate_gnss_signal(num_svs: int, fs: float, nc: int) -> np.ndarray:
    """
    Generate a composite GPS L1 C/A baseband signal from multiple SVs.
    Each SV gets a random code-phase offset and Doppler shift.
    Returns complex IQ array of length nc.
    """
    t = np.arange(nc) / fs
    composite = np.zeros(nc, dtype=complex)

    prns = np.random.choice(range(1, 33), size=num_svs, replace=False)
    for prn in prns:
        ca_code = generate_ca_code(prn)

        # Upsample C/A code to sampling rate
        code_phase = np.random.uniform(0, 1023)
        chip_indices = (np.arange(nc) * F_CODE / fs + code_phase) % 1023
        code_sampled = ca_code[chip_indices.astype(int)]

        # Random Doppler shift (typical: +/- 5 kHz)
        doppler = np.random.uniform(-5000, 5000)
        carrier = np.exp(1j * 2 * np.pi * doppler * t)

        # Random amplitude (satellites at different elevations)
        amplitude = np.random.uniform(0.5, 1.5)

        # Random initial phase
        phase = np.random.uniform(0, 2 * np.pi)

        composite += amplitude * code_sampled * carrier * np.exp(1j * phase)

    return composite


# ---------------------------------------------------------------------------
# Channel model (simplified Rayleigh + delay)
# ---------------------------------------------------------------------------
def apply_channel(signal: np.ndarray, fs: float) -> np.ndarray:
    """Apply a simplified multipath channel to the signal."""
    output = signal.copy()
    num_paths = np.random.randint(1, 4)  # 1-3 multipath components
    for _ in range(num_paths):
        delay_samples = np.random.randint(1, int(0.001 * fs))  # up to 1 ms
        attenuation = np.random.uniform(0.05, 0.3)
        phase_shift = np.random.uniform(0, 2 * np.pi)
        delayed = np.roll(signal, delay_samples) * attenuation * np.exp(1j * phase_shift)
        output += delayed
    return output


# ---------------------------------------------------------------------------
# AWGN addition at a specified CNR
# ---------------------------------------------------------------------------
def add_awgn(signal: np.ndarray, cnr_dbhz: float, fs: float) -> np.ndarray:
    """
    Add AWGN to achieve approximately the specified C/N0.
    CNR_dBHz = 10*log10(P_signal / N0), where N0 is noise PSD.
    """
    # Normalize signal power to 1
    sig_power = np.mean(np.abs(signal) ** 2)
    signal = signal / np.sqrt(sig_power)

    # Noise power from CNR: N0 = P_signal / (10^(CNR/10))
    # Total noise power in bandwidth fs: P_noise = N0 * fs
    cnr_linear = 10 ** (cnr_dbhz / 10)
    noise_power = fs / cnr_linear  # per the signal power = 1 normalization
    noise_std = np.sqrt(noise_power / 2)  # per real/imag component

    noise = noise_std * (np.random.randn(len(signal)) +
                         1j * np.random.randn(len(signal)))
    return signal + noise


# ---------------------------------------------------------------------------
# Jammer signal generators
# ---------------------------------------------------------------------------
def generate_jammer(jam_type: int, fs: float, nc: int) -> np.ndarray:
    """
    Generate a baseband jammer signal of the specified type.

    Types (matching MATLAB JammerType_Vec cases):
        1 - NoJam       → returns zeros
        2 - Single AM   → amplitude-modulated carrier
        3 - Single Chirp→ linear frequency sweep
        5 - Single FM   → frequency-modulated carrier
        9 - DME/Pulse   → Gaussian pulse pairs (DME-like)
        10 - Narrowband → continuous wave (single tone)
    """
    t = np.arange(nc) / fs

    if jam_type == 1:  # No jammer
        return np.zeros(nc, dtype=complex)

    elif jam_type == 2:  # AM jammer
        f_carrier = np.random.uniform(-fs / 4, fs / 4)
        f_mod = np.random.uniform(100, 5000)        # AM modulation freq
        mod_depth = np.random.uniform(0.3, 1.0)
        envelope = 1 + mod_depth * np.sin(2 * np.pi * f_mod * t)
        carrier = np.exp(1j * 2 * np.pi * f_carrier * t)
        return envelope * carrier

    elif jam_type == 3:  # Chirp jammer
        bw = np.random.uniform(1e6, fs / 2)         # Chirp bandwidth
        f_start = np.random.uniform(-bw / 2, 0)
        sweep_rate = bw / (nc / fs)                  # Hz/s
        phase = 2 * np.pi * (f_start * t + 0.5 * sweep_rate * t ** 2)
        return np.exp(1j * phase)

    elif jam_type == 5:  # FM jammer
        f_carrier = np.random.uniform(-fs / 4, fs / 4)
        f_mod = np.random.uniform(100, 10000)
        freq_dev = np.random.uniform(1e4, fs / 4)   # Frequency deviation
        phase = 2 * np.pi * f_carrier * t + (freq_dev / f_mod) * np.sin(2 * np.pi * f_mod * t)
        return np.exp(1j * phase)

    elif jam_type == 9:  # DME / Pulse jammer
        # DME: Gaussian pulse pairs, ~3.5 μs width, 12 μs spacing, ~2700 ppps
        pulse_rate = np.random.uniform(2000, 3000)   # Pulse pairs per second
        pulse_width = 3.5e-6                          # seconds
        pair_spacing = 12e-6                          # seconds
        signal = np.zeros(nc, dtype=complex)
        num_pulses = int(pulse_rate * nc / fs)
        pulse_positions = np.sort(np.random.choice(nc, size=num_pulses, replace=False))
        sigma = pulse_width * fs / 2.355  # Gaussian width in samples
        for pos in pulse_positions:
            for offset in [0, int(pair_spacing * fs)]:
                p = pos + offset
                if p < nc:
                    indices = np.arange(max(0, p - int(4 * sigma)),
                                        min(nc, p + int(4 * sigma)))
                    signal[indices] += np.exp(-0.5 * ((indices - p) / sigma) ** 2)
        # Add a carrier offset
        f_carrier = np.random.uniform(-fs / 4, fs / 4)
        signal = signal * np.exp(1j * 2 * np.pi * f_carrier * t[:len(signal)])
        return signal

    elif jam_type == 10:  # Narrowband (CW)
        f_cw = np.random.uniform(-fs / 4, fs / 4)
        phase0 = np.random.uniform(0, 2 * np.pi)
        return np.exp(1j * (2 * np.pi * f_cw * t + phase0))

    else:
        raise ValueError(f"Unknown jammer type: {jam_type}")


def add_jammer_at_jsr(gnss_signal: np.ndarray, jammer_signal: np.ndarray,
                      jsr_db: float) -> np.ndarray:
    """
    Combine GNSS and jammer signals at the specified JSR (dB).
    JSR = 10*log10(P_jammer / P_signal)
    """
    if np.all(jammer_signal == 0):
        return gnss_signal

    p_gnss = np.mean(np.abs(gnss_signal) ** 2)
    p_jam = np.mean(np.abs(jammer_signal) ** 2)

    # Scale jammer to achieve desired JSR relative to GNSS signal
    jsr_linear = 10 ** (jsr_db / 10)
    scale = np.sqrt(jsr_linear * p_gnss / p_jam)

    return gnss_signal + scale * jammer_signal


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------
CLASS_MAP = {
    1:  "NoJam",
    2:  "SingleAM",
    3:  "SingleChirp",
    5:  "SingleFM",
    9:  "DME",
    10: "NB",
}

JAMMER_TYPES = [1, 2, 3, 5, 9, 10]  # Matching MATLAB JammerType_Vec


def generate_dataset(output_dir: str, num_samples: int, is_training: bool,
                     fs: float = FS, nc: int = NC, num_svs: int = NUM_SVS,
                     save_format: str = "npy"):
    """
    Generate the full Swinney-equivalent dataset.

    Args:
        output_dir:   Root output directory
        num_samples:  Number of samples per class
        is_training:  True for training set, False for test set
        fs:           Sampling frequency (Hz)
        nc:           Samples per segment
        num_svs:      Number of GPS satellites to simulate
        save_format:  "npy" for numpy, "mat" for MATLAB compatibility
    """
    split = "training" if is_training else "testing"
    base_dir = Path(output_dir) / f"Image_{split}_database"

    # Create class directories
    for class_name in CLASS_MAP.values():
        (base_dir / class_name).mkdir(parents=True, exist_ok=True)

    total = num_samples * len(JAMMER_TYPES)
    count = 0

    for sample_idx in range(1, num_samples + 1):
        # Random CNR and JSR for this sample (matching MATLAB: uniform draws)
        cnr_dbhz = np.random.uniform(25, 50)
        jsr_db = np.random.uniform(40, 80)

        # Generate base GNSS signal
        gnss_signal = generate_gnss_signal(num_svs, fs, nc)
        gnss_with_channel = apply_channel(gnss_signal, fs)

        for jam_type in JAMMER_TYPES:
            # Generate jammer
            jammer = generate_jammer(jam_type, fs, nc)

            if jam_type == 1:  # No jammer — just GNSS + AWGN
                combined = add_awgn(gnss_with_channel, cnr_dbhz, fs)
            else:
                # Combine GNSS + jammer at JSR, then add AWGN
                combined_no_noise = add_jammer_at_jsr(gnss_with_channel, jammer, jsr_db)
                combined = add_awgn(combined_no_noise, cnr_dbhz, fs)

            # Save
            class_name = CLASS_MAP[jam_type]
            prefix = "Training" if is_training else "Testing"
            fname = f"{prefix}_raw_{sample_idx}"

            if save_format == "mat":
                filepath = base_dir / class_name / f"{fname}.mat"
                savemat(str(filepath), {"GNSS_plus_Jammer_awgn": combined})
            else:
                filepath = base_dir / class_name / f"{fname}.npy"
                np.save(str(filepath), combined)

            count += 1
            if count % 100 == 0 or count == total:
                print(f"  [{count}/{total}] ({100 * count / total:.1f}%)")

    print(f"\nDone. {count} files saved to {base_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic GNSS + jamming IQ dataset "
                    "(Python translation of Swinney/Morales Ferre)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Root output directory")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Samples per class (default: 1000 training, 250 test)")
    parser.add_argument("--training", action="store_true",
                        help="Generate training data (default: test data)")
    parser.add_argument("--format", choices=["npy", "mat"], default="npy",
                        help="Output format: npy (numpy) or mat (MATLAB)")
    parser.add_argument("--fs", type=float, default=FS,
                        help=f"Sampling frequency in Hz (default: {FS:.0f})")
    parser.add_argument("--nc", type=int, default=NC,
                        help=f"Samples per segment (default: {NC})")

    args = parser.parse_args()
    generate_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        is_training=args.training,
        fs=args.fs,
        nc=args.nc,
        save_format=args.format,
    )
