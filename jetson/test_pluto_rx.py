#!/usr/bin/env python3
"""
test_pluto_rx.py  —  PlutoSDR Reception Test via libiio (no GNU Radio)

Configures the PlutoSDR directly through the IIO kernel driver, tunes to
GPS L1 (1575.42 MHz) at 5 MSPS, collects 10 seconds of IQ data, then:
  1. Reports mean power and flags clipping / antenna issues.
  2. Renders an ASCII PSD bar chart to the terminal (no display needed).
  3. Saves a 1-second IQ snapshot (.npy, complex64) for offline inspection.

This script is the replacement for test_pluto_gnuradio.py after it was
discovered that the Jetson Nano's GNU Radio installation is Python 2.7 only.
libiio (pylibiio 0.25) provides a clean Python 3 IIO interface.

Usage:
    python3 test_pluto_rx.py [--uri ip:192.168.2.1] [--gain 40]

IIO device topology (AD9361 / AD9363):
    ad9361-phy          — control device (frequency, gain, bandwidth)
    cf-ad9361-lpc       — streaming device (raw I/Q DMA)
    cf-ad9361-dds-core  — TX DDS (not used here)
"""

import argparse
import sys
import time
import numpy as np

# ---------------------------------------------------------------------------
# Configuration — must match training pipeline exactly
# ---------------------------------------------------------------------------
GPS_L1_HZ       = 1_575_420_000   # GPS L1 centre frequency, Hz
SAMPLE_RATE_SPS = 5_000_000       # 5 MSPS — matches OAKBAT / Swinney datasets
RF_BANDWIDTH_HZ = 5_000_000       # AD9363 minimum is ~1 MHz; 5 MHz matches SR
BUFFER_SAMPLES  = 1 << 15         # 32768 samples per IIO DMA buffer (~6.5 ms)
COLLECT_SEC     = 10              # seconds of IQ data to collect
SNAP_SEC        = 1               # seconds to save as .npy snapshot

# AD9361 12-bit ADC stored in 16-bit words.
# Full-scale is ±2047 (2^11 - 1); divide by 2048 for ±1.0 normalisation.
ADC_SCALE       = 2048.0

# ASCII spectrum settings
SPEC_BINS       = 48
SPEC_ROWS       = 12


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="PlutoSDR IIO reception test — GPS L1 (no GNU Radio)")
    p.add_argument("--uri",  default="ip:192.168.2.1",
                   help="IIO context URI  (ip:192.168.2.1  or  usb:1.6.5)")
    p.add_argument("--freq", type=int, default=GPS_L1_HZ,
                   help="Centre frequency Hz (default: GPS L1 = 1575420000)")
    p.add_argument("--samp-rate", type=int, default=SAMPLE_RATE_SPS,
                   help="Sample rate SPS (default: 5000000)")
    p.add_argument("--bandwidth", type=int, default=RF_BANDWIDTH_HZ,
                   help="RF bandwidth Hz (default: 5000000)")
    p.add_argument("--gain-mode", default="slow_attack",
                   choices=["slow_attack", "fast_attack", "manual", "hybrid"],
                   help="AGC mode (default: slow_attack)")
    p.add_argument("--gain", type=float, default=40.0,
                   help="Manual gain dB, 0–71 (only used when --gain-mode manual)")
    p.add_argument("--duration", type=float, default=COLLECT_SEC,
                   help="Collection duration in seconds (default: 10)")
    p.add_argument("--snap-file", default="pluto_l1_snapshot.npy",
                   help="Output filename for 1-second IQ snapshot")
    return p.parse_args()


# ---------------------------------------------------------------------------
# IIO helper: open context with a clear error message
# ---------------------------------------------------------------------------
def open_context(uri: str):
    try:
        import iio
    except ImportError:
        print("[FAIL] pylibiio not installed.  Run: pip3 install pylibiio")
        sys.exit(1)

    print(f"[INFO] Opening IIO context at {uri} ...")
    try:
        ctx = iio.Context(uri)
    except Exception as e:
        print(f"[FAIL] Cannot open IIO context at {uri}: {e}")
        print()
        print("  Troubleshooting:")
        print("    iio_info -s                 # scan for visible devices")
        print("    ping 192.168.2.1            # check USB RNDIS link")
        print("    dmesg | tail -20            # check for USB errors")
        sys.exit(1)

    # Print connected devices for confirmation
    print(f"[INFO] Context opened. Devices:")
    for dev in ctx.devices:
        print(f"         {dev.name or dev.id}")
    print()
    return ctx


# ---------------------------------------------------------------------------
# IIO helper: configure AD9361 PHY (frequency, sample rate, gain)
# ---------------------------------------------------------------------------
def configure_phy(ctx, freq_hz: int, samp_rate: int, bandwidth: int,
                  gain_mode: str, gain_db: float):
    """
    Write RF parameters to the ad9361-phy control device.
    All attributes are set as strings (IIO ABI requirement).
    """
    phy = ctx.find_device("ad9361-phy")
    if phy is None:
        print("[FAIL] ad9361-phy device not found.")
        print("       Is this an AD9361/AD9363-based PlutoSDR?")
        sys.exit(1)

    # RX voltage channel (voltage0, input direction)
    rx_ch = phy.find_channel("voltage0", False)   # False = input (RX)
    if rx_ch is None:
        print("[FAIL] ad9361-phy voltage0 RX channel not found.")
        sys.exit(1)

    print(f"[INFO] Configuring PHY:")
    print(f"         Frequency   : {freq_hz / 1e6:.3f} MHz")
    print(f"         Sample rate : {samp_rate / 1e6:.1f} MSPS")
    print(f"         RF bandwidth: {bandwidth / 1e6:.1f} MHz")
    print(f"         Gain mode   : {gain_mode}")

    # RF bandwidth and sample rate
    rx_ch.attrs["rf_bandwidth"].value        = str(bandwidth)
    rx_ch.attrs["sampling_frequency"].value  = str(samp_rate)

    # Gain control
    rx_ch.attrs["gain_control_mode"].value = gain_mode
    if gain_mode == "manual":
        # Clamp to AD9361 supported range
        gain_clamped = max(0.0, min(71.0, gain_db))
        rx_ch.attrs["hardwaregain"].value = str(gain_clamped)
        print(f"         Manual gain : {gain_clamped:.0f} dB")
    else:
        # Read back whatever the AGC settled on (informational)
        try:
            agc_gain = rx_ch.attrs["hardwaregain"].value
            print(f"         AGC gain    : {agc_gain} dB (current)")
        except Exception:
            pass

    # LO frequency — altvoltage0 is the RX local oscillator
    lo_ch = phy.find_channel("altvoltage0", True)   # True = output (LO)
    if lo_ch is None:
        print("[FAIL] ad9361-phy altvoltage0 LO channel not found.")
        sys.exit(1)
    lo_ch.attrs["frequency"].value = str(freq_hz)

    print()


# ---------------------------------------------------------------------------
# IIO helper: enable streaming channels and create buffer
# ---------------------------------------------------------------------------
def setup_streaming(ctx, buffer_samples: int):
    """
    Enable I and Q channels on the cf-ad9361-lpc streaming device
    and allocate a DMA buffer.

    Buffer layout (AD9361 default):
        interleaved int16 pairs: [I0, Q0, I1, Q1, ...]
    """
    rx_dev = ctx.find_device("cf-ad9361-lpc")
    if rx_dev is None:
        print("[FAIL] cf-ad9361-lpc streaming device not found.")
        sys.exit(1)

    import iio
    i_ch = rx_dev.find_channel("voltage0", False)
    q_ch = rx_dev.find_channel("voltage1", False)

    if i_ch is None or q_ch is None:
        print("[FAIL] I/Q streaming channels not found on cf-ad9361-lpc.")
        sys.exit(1)

    i_ch.enabled = True
    q_ch.enabled = True

    buf = iio.Buffer(rx_dev, buffer_samples, False)   # non-cyclic
    print(f"[INFO] DMA buffer allocated: {buffer_samples} samples "
          f"({buffer_samples / SAMPLE_RATE_SPS * 1000:.1f} ms)")
    print()
    return buf


# ---------------------------------------------------------------------------
# Collect IQ samples into a numpy array
# ---------------------------------------------------------------------------
def collect_iq(buf, total_samples: int, samp_rate: int) -> np.ndarray:
    collected = []
    t0 = time.time()
    timeout = total_samples / samp_rate + 10.0

    print(f"[INFO] Collecting {total_samples:,} samples "
          f"({total_samples / samp_rate:.1f} s)...")

    while len(collected) < total_samples:
        if time.time() - t0 > timeout:
            print(f"\n[WARN] Timeout — collected {len(collected):,} / {total_samples:,} samples.")
            break

        buf.refill()
        raw = buf.read()

        # Parse interleaved int16 → complex float32
        samples = np.frombuffer(raw, dtype=np.int16)
        i_raw   = samples[0::2].astype(np.float32)
        q_raw   = samples[1::2].astype(np.float32)
        iq      = (i_raw + 1j * q_raw) / ADC_SCALE

        remaining = total_samples - len(collected)
        collected.append(iq[:remaining])

        pct = 100 * len(collected) / total_samples   # approximate
        elapsed = time.time() - t0
        print(f"\r  {sum(len(c) for c in collected):>9,} / {total_samples:,} samples  "
              f"({elapsed:.1f}s)", end="", flush=True)

    print()  # newline after \r
    return np.concatenate(collected)


# ---------------------------------------------------------------------------
# Diagnostics: power assessment
# ---------------------------------------------------------------------------
def assess_power(iq: np.ndarray):
    mean_pwr_dbfs = 10  * np.log10(np.mean(np.abs(iq) ** 2)  + 1e-30)
    peak_dbfs     = 20  * np.log10(np.max(np.abs(iq))         + 1e-30)
    dc_pwr        = np.abs(np.mean(iq)) ** 2
    sig_pwr       = np.mean(np.abs(iq) ** 2)
    dc_ratio_db   = 10  * np.log10(dc_pwr / (sig_pwr + 1e-30) + 1e-30)

    print("--- Power Assessment ---")
    print(f"  Mean power  : {mean_pwr_dbfs:+.1f} dBFS")
    print(f"  Peak power  : {peak_dbfs:+.1f} dBFS")
    print(f"  DC offset   : {dc_ratio_db:.1f} dB relative to signal")
    print()

    # Open-sky GPS L1 at ~40 dB gain typically lands −55 to −25 dBFS.
    if mean_pwr_dbfs < -70:
        print("  ⚠  Power very low.")
        print("     • Check antenna connection and point toward open sky.")
        print("     • Try increasing gain: --gain-mode manual --gain 55")
        print("     • GPS L1 C/A is ~−130 dBm at antenna; you need ~40 dB LNA gain.")
    elif mean_pwr_dbfs > -3:
        print("  ⚠  Near ADC clipping — reduce gain.")
        print("     Try: --gain-mode manual --gain 20")
    else:
        print("  ✓  Power level reasonable for GPS L1 reception.")

    if dc_ratio_db > -20:
        print(f"  ⚠  DC offset significant ({dc_ratio_db:.1f} dB).")
        print("     Verify ad9361-phy bb_dc and rf_dc corrections are active.")
        print("     Check: iio_attr -u ip:192.168.2.1 -D ad9361-phy bb_dc_offset_tracking_en")
    else:
        print(f"  ✓  DC offset acceptable ({dc_ratio_db:.1f} dB).")
    print()


# ---------------------------------------------------------------------------
# Diagnostics: ASCII PSD bar chart
# ---------------------------------------------------------------------------
def ascii_spectrum(iq: np.ndarray, fs: float,
                   n_bins: int = SPEC_BINS, n_rows: int = SPEC_ROWS):
    chunk   = 4096
    n_chunks = min(len(iq) // chunk, 512)
    if n_chunks == 0:
        print("  [WARN] Not enough samples for spectrum estimate.")
        return

    window  = np.hanning(chunk)
    psd     = np.zeros(chunk)
    for i in range(n_chunks):
        seg = iq[i * chunk:(i + 1) * chunk]
        psd += np.abs(np.fft.fftshift(np.fft.fft(seg * window))) ** 2
    psd    /= n_chunks
    psd_db  = 10 * np.log10(psd + 1e-30)

    # Downsample to n_bins
    edges  = np.linspace(0, len(psd_db), n_bins + 1, dtype=int)
    binned = np.array([psd_db[edges[k]:edges[k + 1]].mean() for k in range(n_bins)])

    lo, hi = binned.min(), binned.max()
    norm   = (binned - lo) / (hi - lo + 1e-10)

    bw_mhz = fs / 1e6
    print("--- ASCII Power Spectral Density ---")
    print(f"  Centre: {GPS_L1_HZ / 1e6:.3f} MHz   "
          f"Span: ±{bw_mhz / 2:.1f} MHz   "
          f"Range: {lo:.1f} to {hi:.1f} dB (relative)")
    print()

    bar = " ▁▂▃▄▅▆▇█"
    for row in range(n_rows, 0, -1):
        thr  = row / n_rows
        line = "  |"
        for v in norm:
            line += bar[min(8, int(v * 8))] if v >= thr else " "
        line += "|"
        if row == n_rows:
            line += f"  {hi:.1f} dB"
        elif row == 1:
            line += f"  {lo:.1f} dB"
        print(line)

    print("  +" + "─" * n_bins + "+")

    # Freq axis labels: −BW/2, 0, +BW/2
    left  = f"−{bw_mhz/2:.1f}"
    mid   = "0"
    right = f"+{bw_mhz/2:.1f}"
    axis  = "  " + left + " " * (n_bins // 2 - len(left) - 1) + \
            mid.center(3) + \
            " " * (n_bins // 2 - len(right) - 1) + right
    print(axis)
    print(f"  {'MHz (offset from centre)':>{n_bins + 2}}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print("=" * 60)
    print("  PlutoSDR IIO Reception Test — GPS L1  (no GNU Radio)")
    print("=" * 60)
    print(f"  URI         : {args.uri}")
    print(f"  Frequency   : {args.freq / 1e6:.3f} MHz")
    print(f"  Sample rate : {args.samp_rate / 1e6:.1f} MSPS")
    print(f"  RF bandwidth: {args.bandwidth / 1e6:.1f} MHz")
    print(f"  Gain mode   : {args.gain_mode}"
          + (f"  ({args.gain:.0f} dB)" if args.gain_mode == "manual" else ""))
    print(f"  Duration    : {args.duration} s")
    print()

    ctx = open_context(args.uri)

    configure_phy(
        ctx,
        freq_hz   = args.freq,
        samp_rate = args.samp_rate,
        bandwidth = args.bandwidth,
        gain_mode = args.gain_mode,
        gain_db   = args.gain,
    )

    buf = setup_streaming(ctx, BUFFER_SAMPLES)

    # Brief settle time for AGC / LO to stabilise
    print("[INFO] Waiting 1 s for LO and AGC to settle...")
    time.sleep(1.0)

    total_samples = int(args.samp_rate * args.duration)
    iq = collect_iq(buf, total_samples, args.samp_rate)

    print(f"[INFO] Collected {len(iq):,} samples ({len(iq) / args.samp_rate:.2f} s).")
    print()

    assess_power(iq)
    ascii_spectrum(iq, args.samp_rate)

    # Save 1-second snapshot for offline inspection / pipeline debugging
    snap_samples = min(len(iq), args.samp_rate * SNAP_SEC)
    snap = iq[:snap_samples].astype(np.complex64)
    np.save(args.snap_file, snap)
    print(f"[INFO] Snapshot saved → {args.snap_file}")
    print(f"       Shape: {snap.shape}  dtype: {snap.dtype}  "
          f"({snap.nbytes / 1e6:.1f} MB)")
    print()
    print("  Next step:")
    print(f"    python3 demo_inference.py --snap {args.snap_file} --model model.trt")
    print("=" * 60)


if __name__ == "__main__":
    main()
