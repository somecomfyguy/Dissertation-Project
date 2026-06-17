#!/usr/bin/env python3
"""
test_pluto_gnuradio.py  —  Headless PlutoSDR Reception Test at GPS L1

Tunes the PlutoSDR to GPS L1 (1575.42 MHz) at 5 MSPS, collects 10 seconds of
IQ data, then:
  1. Prints mean received power across the band.
  2. Renders an ASCII power-spectrum bar chart (no display needed).
  3. Saves a 1-second IQ snapshot (.npy) for offline inspection.
  4. Reports whether the power level is in a sensible range for GPS L1.

Usage:
    python3 test_pluto_gnuradio.py [--uri ip:192.168.2.1] [--gain 40]

If gr-iio is not available, falls back to SoapySDR (osmosdr) with a note.
"""

import argparse
import sys
import time
import numpy as np

# ---------------------------------------------------------------------------
# Configuration — must match training pipeline
# ---------------------------------------------------------------------------
GPS_L1_HZ       = 1_575_420_000     # GPS L1 centre frequency
SAMPLE_RATE_SPS = 5_000_000         # 5 MSPS — matches OAKBAT/Swinney
RF_BANDWIDTH_HZ = 5_000_000         # 5 MHz passband
BUFFER_SIZE     = 0x8000            # 32768 samples per IIO buffer
COLLECT_SEC     = 10                # seconds to collect
SNAP_SEC        = 1                 # seconds to save as .npy snapshot
SNAP_SAMPLES    = SAMPLE_RATE_SPS * SNAP_SEC  # 5M samples

# ASCII spectrum settings
SPECTRUM_BINS   = 32               # horizontal resolution of bar chart
SPECTRUM_ROWS   = 12               # vertical rows

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Headless PlutoSDR GPS L1 reception test")
    p.add_argument("--uri",  default="ip:192.168.2.1",
                   help="IIO context URI (e.g. ip:192.168.2.1 or usb:1.6.5)")
    p.add_argument("--gain", type=float, default=40.0,
                   help="RX gain in dB (0–71); use slow_attack AGC if 0")
    p.add_argument("--freq", type=float, default=GPS_L1_HZ,
                   help="Centre frequency in Hz (default: GPS L1 = 1575.42 MHz)")
    p.add_argument("--samp-rate", type=float, default=SAMPLE_RATE_SPS,
                   help="Sample rate in SPS (default: 5000000)")
    p.add_argument("--snap-file", default="pluto_l1_snapshot.npy",
                   help="Output filename for 1-second IQ snapshot")
    p.add_argument("--backend", choices=["grIio", "soapy"], default="grIio",
                   help="GNU Radio source backend to use")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Backend: gr-iio PlutoSDR source
# ---------------------------------------------------------------------------
def try_griio(args):
    """Attempt to build a GNU Radio flowgraph using gr-iio PlutoSDR block."""
    try:
        from gnuradio import gr, blocks
        from gnuradio import iio
    except ImportError as e:
        return None, f"gr-iio not available: {e}"

    class _Sink(gr.sync_block):
        """Custom sink that accumulates samples into a list."""
        def __init__(self, max_samples):
            gr.sync_block.__init__(self, "AccumulatorSink",
                                   in_sig=[np.complex64], out_sig=[])
            self.max_samples = max_samples
            self.buffer = []
            self.done = False

        def work(self, input_items, output_items):
            if self.done:
                return 0
            chunk = input_items[0]
            remaining = self.max_samples - len(self.buffer)
            self.buffer.extend(chunk[:remaining].tolist())
            if len(self.buffer) >= self.max_samples:
                self.done = True
            return len(chunk)

    gain_mode = "slow_attack" if args.gain <= 0 else "manual"
    gain_val  = max(0, min(71, int(args.gain)))

    try:
        tb = gr.top_block()

        # PlutoSDR source — gr-iio API (GNU Radio 3.8 / gr-iio 0.3+)
        # Signature: pluto_source(uri, freq, samp_rate, bandwidth,
        #                         buffer_size, quadrature, rfdc, bbdc,
        #                         gain_mode, gain, filter, auto_filter)
        src = iio.pluto_source(
            args.uri,
            int(args.freq),
            int(args.samp_rate),
            RF_BANDWIDTH_HZ,
            BUFFER_SIZE,
            True,        # quadrature
            True,        # RF DC correction
            True,        # BB DC correction
            gain_mode,
            gain_val,
            "",          # custom filter (empty = use default)
            True,        # auto_filter
        )

        total_samples = int(args.samp_rate * COLLECT_SEC)
        sink = _Sink(total_samples)
        tb.connect(src, sink)
        return tb, sink, None

    except Exception as e:
        return None, None, f"gr-iio PlutoSDR source failed: {e}"


# ---------------------------------------------------------------------------
# Backend: SoapySDR / osmosdr fallback
# ---------------------------------------------------------------------------
def try_soapy(args):
    try:
        from gnuradio import gr, blocks
        import osmosdr
    except ImportError as e:
        return None, None, f"osmosdr not available: {e}"

    class _Sink(gr.sync_block):
        def __init__(self, max_samples):
            gr.sync_block.__init__(self, "AccumulatorSink",
                                   in_sig=[np.complex64], out_sig=[])
            self.max_samples = max_samples
            self.buffer = []
            self.done = False

        def work(self, input_items, output_items):
            if self.done:
                return 0
            chunk = input_items[0]
            remaining = self.max_samples - len(self.buffer)
            self.buffer.extend(chunk[:remaining].tolist())
            if len(self.buffer) >= self.max_samples:
                self.done = True
            return len(chunk)

    try:
        tb = gr.top_block()
        # SoapySDR driver string for PlutoSDR clone
        driver_str = f"soapy=0,driver=plutosdr,uri={args.uri}"
        src = osmosdr.source(args=driver_str)
        src.set_sample_rate(args.samp_rate)
        src.set_center_freq(args.freq)
        src.set_freq_corr(0)
        src.set_gain_mode(args.gain <= 0, 0)  # AGC if gain=0
        src.set_gain(max(0, args.gain), 0)
        src.set_bandwidth(RF_BANDWIDTH_HZ, 0)

        total_samples = int(args.samp_rate * COLLECT_SEC)
        sink = _Sink(total_samples)
        tb.connect(src, sink)
        return tb, sink, None
    except Exception as e:
        return None, None, f"SoapySDR source failed: {e}"


# ---------------------------------------------------------------------------
# Diagnostics: ASCII power spectrum
# ---------------------------------------------------------------------------
def ascii_spectrum(iq: np.ndarray, fs: float, n_bins: int = SPECTRUM_BINS,
                   n_rows: int = SPECTRUM_ROWS):
    """
    Print an ASCII bar chart of the power spectral density.
    Frequencies run from -fs/2 to +fs/2 (DC in the centre).
    """
    # Welch-style PSD: average over chunks
    chunk = 4096
    n_chunks = len(iq) // chunk
    psd = np.zeros(chunk, dtype=np.float64)
    for i in range(n_chunks):
        seg = iq[i * chunk:(i + 1) * chunk]
        psd += np.abs(np.fft.fftshift(np.fft.fft(seg * np.hanning(chunk)))) ** 2
    psd /= n_chunks
    psd_db = 10 * np.log10(psd + 1e-30)

    # Downsample to n_bins
    bin_edges = np.linspace(0, len(psd_db), n_bins + 1, dtype=int)
    binned = np.array([psd_db[bin_edges[i]:bin_edges[i + 1]].mean()
                       for i in range(n_bins)])

    lo, hi = binned.min(), binned.max()
    norm = (binned - lo) / (hi - lo + 1e-10)  # 0..1

    bw_mhz = fs / 1e6
    freq_labels = [f"{-bw_mhz/2 + bw_mhz * i / n_bins:.1f}" for i in range(n_bins)]

    print("\n  Power Spectral Density  (relative, dB)")
    print(f"  Freq range: {-bw_mhz/2:.1f} to +{bw_mhz/2:.1f} MHz around {GPS_L1_HZ/1e6:.3f} MHz")
    print()

    bar_chars = " ▁▂▃▄▅▆▇█"
    for row in range(n_rows, 0, -1):
        threshold = row / n_rows
        line = "  |"
        for v in norm:
            if v >= threshold:
                line += bar_chars[min(8, int(v * 8))]
            else:
                line += " "
        line += "|"
        if row == n_rows:
            line += f"  {hi:.1f} dB"
        elif row == 1:
            line += f"  {lo:.1f} dB"
        print(line)

    print("  +" + "-" * n_bins + "+")
    # Frequency axis: print first, middle, last labels
    axis = " " * 3
    axis += freq_labels[0].ljust(n_bins // 2)
    axis += "0.0".center(1)
    axis += freq_labels[-1].rjust(n_bins // 2 - 2)
    print(axis)
    print(f"  {'MHz (relative to centre)':>{n_bins + 2}}")
    print()


# ---------------------------------------------------------------------------
# Power assessment
# ---------------------------------------------------------------------------
def assess_power(iq: np.ndarray, samp_rate: float):
    mean_pwr_dbfs = 10 * np.log10(np.mean(np.abs(iq) ** 2) + 1e-30)
    peak_dbfs     = 20 * np.log10(np.max(np.abs(iq)) + 1e-30)

    print(f"  Mean power : {mean_pwr_dbfs:+.1f} dBFS")
    print(f"  Peak power : {peak_dbfs:+.1f} dBFS")
    print()

    # Expected range: GPS L1 at the antenna is ~-130 dBm.
    # With ~40 dB LNA gain and PlutoSDR's own noise, a typical
    # open-sky reading lands around -50 to -30 dBFS.
    # Values below -70 dBFS likely mean: no signal / antenna not connected.
    # Values near 0 dBFS indicate clipping — reduce gain.
    if mean_pwr_dbfs < -70:
        print("  ⚠  Power very low — check antenna connection and pointing.")
        print("     Typical open-sky GPS L1 lands around −50 to −30 dBFS at 40 dB gain.")
    elif mean_pwr_dbfs > -3:
        print("  ⚠  Possible ADC clipping — reduce gain (--gain 20).")
    else:
        print("  ✓  Power level looks reasonable for GPS L1 reception.")

    # DC spike check (PlutoSDR clones sometimes have bad DC offset correction)
    dc_power = np.abs(np.mean(iq)) ** 2
    sig_power = np.mean(np.abs(iq) ** 2)
    dc_ratio_db = 10 * np.log10(dc_power / (sig_power + 1e-30) + 1e-30)
    if dc_ratio_db > -20:
        print(f"  ⚠  DC offset detected ({dc_ratio_db:.1f} dB relative) — enable BB DC correction.")
    else:
        print(f"  ✓  DC offset acceptable ({dc_ratio_db:.1f} dB relative).")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print("=" * 60)
    print("  PlutoSDR Reception Test — GPS L1")
    print("=" * 60)
    print(f"  URI         : {args.uri}")
    print(f"  Frequency   : {args.freq / 1e6:.3f} MHz (GPS L1 = 1575.420 MHz)")
    print(f"  Sample rate : {args.samp_rate / 1e6:.1f} MSPS")
    print(f"  Gain        : {args.gain:.0f} dB ({'manual' if args.gain > 0 else 'AGC slow_attack'})")
    print(f"  Duration    : {COLLECT_SEC} s")
    print()

    # Build flowgraph
    tb = sink = None
    if args.backend == "grIio":
        result = try_griio(args)
        tb, sink, err = result
        if err:
            print(f"[WARN] gr-iio failed ({err}) — trying SoapySDR fallback...")
            tb, sink, err2 = try_soapy(args)
            if err2:
                print(f"[FAIL] SoapySDR also failed: {err2}")
                print("       Ensure either gr-iio or osmosdr is installed and PlutoSDR is connected.")
                sys.exit(1)
            print("[INFO] Using SoapySDR/osmosdr backend.")
    else:
        tb, sink, err = try_soapy(args)
        if err:
            print(f"[FAIL] {err}")
            sys.exit(1)

    # Run flowgraph
    total_samples = int(args.samp_rate * COLLECT_SEC)
    print(f"[INFO] Starting reception ({COLLECT_SEC}s / {total_samples:,} samples)...")
    tb.start()

    # Poll until accumulator is full or timeout
    t0 = time.time()
    while not sink.done and (time.time() - t0) < COLLECT_SEC + 5:
        collected = len(sink.buffer)
        elapsed   = time.time() - t0
        pct       = 100 * collected / total_samples
        print(f"\r  Collecting... {collected:>9,} / {total_samples:,} samples  ({pct:.0f}%)  "
              f"[{elapsed:.1f}s]", end="", flush=True)
        time.sleep(0.5)

    tb.stop()
    tb.wait()
    print()  # newline after \r

    iq = np.array(sink.buffer, dtype=np.complex64)
    print(f"[INFO] Collected {len(iq):,} samples ({len(iq) / args.samp_rate:.2f} s).")
    print()

    if len(iq) < SNAP_SAMPLES:
        print(f"[WARN] Got fewer samples than expected ({len(iq)} < {SNAP_SAMPLES}).")
        print("       Reception may be unstable — check USB connection.")

    # Diagnostics
    print("--- Power Assessment ---")
    assess_power(iq, args.samp_rate)

    print("--- ASCII Spectrum ---")
    ascii_spectrum(iq, args.samp_rate)

    # Save snapshot
    snap = iq[:SNAP_SAMPLES].copy()
    np.save(args.snap_file, snap)
    print(f"[INFO] 1-second IQ snapshot saved → {args.snap_file}")
    print(f"       Shape: {snap.shape}, dtype: {snap.dtype}")
    print()
    print("  Next step:")
    print("    gnss-sdr --config_file=gnss_sdr_pluto.conf")
    print("    # or: python3 demo_inference.py --snap", args.snap_file)
    print("=" * 60)


if __name__ == "__main__":
    main()
