#!/usr/bin/env python3
"""Record gap-free interleaved int16 IQ from PlutoSDR to a .raw file."""
import argparse, sys, time
import numpy as np
import iio

GPS_L1_HZ = 1_575_420_000
FS = 5_000_000
BW = 5_000_000
BUF = 1 << 18  # 262144 samples/buffer (~52 ms) — large to avoid drops

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uri", default="ip:192.168.2.1")
    p.add_argument("--duration", type=float, default=600)
    p.add_argument("--out", default="capture_clearsky.raw")
    p.add_argument("--gain-mode", default="slow_attack")
    p.add_argument("--gain", type=float, default=40)
    a = p.parse_args()

    ctx = iio.Context(a.uri)
    phy = ctx.find_device("ad9361-phy")
    rx = phy.find_channel("voltage0", False)
    rx.attrs["rf_bandwidth"].value = str(int(BW))
    rx.attrs["sampling_frequency"].value = str(int(FS))
    rx.attrs["gain_control_mode"].value = a.gain_mode
    if a.gain_mode == "manual":
        rx.attrs["hardwaregain"].value = str(max(0, min(71, a.gain)))
    lo = phy.find_channel("altvoltage0", True)
    lo.attrs["frequency"].value = str(GPS_L1_HZ)

    dev = ctx.find_device("cf-ad9361-lpc")
    ich = dev.find_channel("voltage0", False)
    qch = dev.find_channel("voltage1", False)
    ich.enabled = True
    qch.enabled = True
    buf = iio.Buffer(dev, BUF, False)

    print(f"[rec] Recording {a.duration}s to {a.out} ...")
    print("[rec] Letting AGC settle (5s)...")
    time.sleep(5.0)

    total = int(FS * a.duration)
    written = 0
    t0 = time.time()
    with open(a.out, "wb") as f:
        while written < total:
            buf.refill()
            raw = buf.read()              # interleaved int16 bytes, native format
            f.write(raw)
            written += len(raw) // 4      # 4 bytes per complex sample
            el = time.time() - t0
            print(f"\r  {written:,}/{total:,} samples ({el:.0f}s)", end="", flush=True)
    print(f"\n[rec] Done. {written:,} samples, {written*4/1e6:.0f} MB. AGC gain now:",
          rx.attrs["hardwaregain"].value, "dB")

if __name__ == "__main__":
    main()
