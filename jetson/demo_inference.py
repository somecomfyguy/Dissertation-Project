#!/usr/bin/env python3
"""
demo_inference.py  —  Real-Time GNSS Interference Classification on Jetson Nano

Pipeline (GPU-accelerated):
    PlutoSDR (libiio) -> 20 ms IQ window -> [GPU: STFT + features + TRT] → label

All heavy computation (STFT, feature extraction, normalization, inference)
runs on the Maxwell GPU via PyTorch + TensorRT.  The CPU only handles IIO
buffer reads and console output.

Usage — live RF:
    python3 demo_inference.py --trt model.trt --uri ip:192.168.2.1

Usage — offline snapshot test:
    python3 demo_inference.py --trt model.trt --snap pluto_l1_snapshot.npy

Dependencies:
    pip3 install pylibiio pycuda
    PyTorch 1.10, TensorRT 8.2 (JetPack 4.6)
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from iq_tee import IQTee

# Constants
SAMPLE_RATE      = 5_000_000
WINDOW_SEC       = 0.020
WINDOW_SAMPLES   = int(SAMPLE_RATE * WINDOW_SEC)   # 100 000

# STFT parameters
NPERSEG          = 256
NOVERLAP         = 192
HOP_LENGTH       = NPERSEG - NOVERLAP              # 64
OUTPUT_SIZE      = (128, 128)
LOG_EPSILON      = 1e-10

# PlutoSDR RF parameters
GPS_L1_HZ        = 1_575_420_000
RF_BANDWIDTH_HZ  = 5_000_000
IIO_BUFFER_SIZE  = 1 << 15

N_FEATURES       = 8

# 11-class taxonomy
CLASS_NAMES = [
    "clean",
    "jam_am",
    "jam_chirp",
    "jam_dme",
    "jam_fm",
    "jam_narrowband",
    "spoof_dynamic",
    "spoof_matched_time",
    "spoof_overpowered_gradual",
    "spoof_overpowered_instant",
    "spoof_position_push",
]

SHORT_LABEL = {
    "clean":                    "CLEAN              ",
    "jam_am":                   "JAM  — AM          ",
    "jam_chirp":                "JAM  — Chirp       ",
    "jam_dme":                  "JAM  — DME         ",
    "jam_fm":                   "JAM  — FM          ",
    "jam_narrowband":           "JAM  — Narrowband  ",
    "spoof_dynamic":            "SPOOF — Dynamic    ",
    "spoof_matched_time":       "SPOOF — Matched T  ",
    "spoof_overpowered_gradual":"SPOOF — OvPwr Grad ",
    "spoof_overpowered_instant":"SPOOF — OvPwr Inst ",
    "spoof_position_push":      "SPOOF — Pos Push   ",
}


# Feature normalization (z-score)
class FeatureNormalizer:
    def __init__(self):
        self.mean = None
        self.std  = None
        self._t_mean = None
        self._t_std  = None

    def to_device(self, device):
        if self.mean is not None:
            self._t_mean = torch.from_numpy(self.mean).to(device)
            self._t_std  = torch.from_numpy(self.std).to(device)

    def transform_gpu(self, features_gpu: torch.Tensor) -> torch.Tensor:
        return (features_gpu - self._t_mean) / self._t_std

    @classmethod
    def load(cls, path: str) -> "FeatureNormalizer":
        import json
        with open(path) as f:
            stats = json.load(f)
        norm = cls()
        norm.mean = np.array(stats["mean"], dtype=np.float32)
        norm.std = np.array(stats["std"],  dtype=np.float32)
        return norm


# GPU Preprocessor for DSP
class GPUPreprocessor:
    """
    Runs STFT, feature extraction, and normalization entirely on GPU
    using PyTorch 1.10 + torch.fft.

    The STFT is implemented manually (windowed segments → batched FFT)
    rather than via torch.stft, for maximum compatibility with PyTorch 1.10
    and complex input on CUDA 10.2.
    """

    def __init__(self, device: torch.device, feat_norm: FeatureNormalizer = None):
        self.device = device

        # Pre-compute Hann window on GPU
        self.hann = torch.hann_window(NPERSEG, device=device, dtype=torch.float32)

        # Pre-compute frame indices for manual STFT
        # For 100,000 samples: n_frames = (100000 - 256) / 64 + 1 = 1559
        n_frames = (WINDOW_SAMPLES - NPERSEG) // HOP_LENGTH + 1
        frame_starts = torch.arange(n_frames, device=device) * HOP_LENGTH
        offsets  = torch.arange(NPERSEG, device=device)
        self.frame_idx = frame_starts.unsqueeze(1) + offsets.unsqueeze(0)
        self.n_frames = n_frames

        # Feature normalizer on GPU
        self.feat_norm = feat_norm
        if feat_norm is not None:
            feat_norm.to_device(device)

        self._log2_N = np.log2(WINDOW_SAMPLES)

        print(f"[GPU] Preprocessor initialized on {device}")
        print(f"[GPU] STFT frames: {n_frames} x {NPERSEG}, hop={HOP_LENGTH}")
        print(f"[GPU] Hann window + frame indices pre-allocated on GPU")
        print()


    def _stft_gpu(self, iq: torch.Tensor) -> torch.Tensor:
        """
        Manual STFT via batched FFT.

        Args: 
            torch.Tensor: complex64 tensor (WINDOW_SAMPLES,) on GPU
        Returns: 
            torch.Tensor: complex tensor (n_frames, NPERSEG)
        """
        frames = iq[self.frame_idx]                     # (n_frames, NPERSEG)
        frames = frames * self.hann.unsqueeze(0)        # apply window
        return torch.fft.fft(frames, n=NPERSEG, dim=1)  # batched FFT


    def iq_to_spectrogram(self, iq: torch.Tensor) -> torch.Tensor:
        """
        GPU STFT -> log-magnitude -> resize -> per-image norm -> 3-ch tensor.

        Returns:
            torch.Tensor: float32 tensor with dimensions (1, 3, 128, 128) on CPU
        """
        Zxx = self._stft_gpu(iq)

        # Log-magnitude: 10·log10(|Z|² + ε)
        mag_sq  = Zxx.real ** 2 + Zxx.imag ** 2
        log_mag = 10.0 * torch.log10(mag_sq + LOG_EPSILON)

        # Transpose to (freq, time) to match scipy STFT convention
        log_mag = log_mag.t()

        # Resize to 128x128 via bilinear interpolation
        log_mag = log_mag.unsqueeze(0).unsqueeze(0)    # (1, 1, F, T)
        spec = F.interpolate(log_mag, size=OUTPUT_SIZE,
                             mode='bilinear', align_corners=False)
        spec = spec.squeeze()                           # (128, 128)

        # Per-image min-max normalization
        lo = spec.min()
        hi = spec.max()
        spec = (spec - lo) / (hi - lo + 1e-10)

        # Replicate to 3 channels, add batch dim
        spec = spec.unsqueeze(0).expand(3, -1, -1).unsqueeze(0)

        return spec.contiguous().cpu().numpy().astype(np.float32)


    def compute_features(self, iq: torch.Tensor) -> torch.Tensor:
        """
        GPU feature extraction — 8 features matching training pipeline.
        
        Returns: 
            float32 tensor (8,) on GPU
        """
        N   = iq.shape[0]
        eps = 1e-12

        # [0] mean_power, [1] PAPR
        power = iq.real ** 2 + iq.imag ** 2
        mean_pow = power.mean()
        papr = power.max() / (mean_pow + eps)

        # PSD via single FFT (reused for CAF — no redundant computation)
        fft_iq = torch.fft.fft(iq)
        psd = (fft_iq.real ** 2 + fft_iq.imag ** 2) / N
        psd_sum = psd.sum()
        psd_norm = psd / (psd_sum + eps)

        # [2] spectral kurtosis (Fisher)
        mu = psd.mean()
        diff = psd - mu
        var = (diff ** 2).mean()
        kurt = (diff ** 4).mean() / (var ** 2 + eps) - 3.0

        # [3] spectral skewness
        std = torch.sqrt(var + eps)
        skew = (diff ** 3).mean() / (std ** 3 + eps)

        # [4] spectral flatness (Wiener entropy)
        log_psd = torch.log(psd + eps)
        geom_mean = torch.exp(log_psd.mean())
        arith_mean = psd.mean() + eps
        flatness = geom_mean / arith_mean

        # [5] spectral entropy (normalized Shannon)
        entropy = -(psd_norm * torch.log2(psd_norm + eps)).sum()
        entropy_norm = entropy / self._log2_N

        # [6] instantaneous bandwidth (90%-power containment)
        cumsum = torch.cumsum(psd_norm, dim=0)
        low_idx = torch.searchsorted(cumsum, 0.05)
        high_idx = torch.searchsorted(cumsum, 0.95)
        bw_norm = (high_idx - low_idx).float() / N

        # [7] CAF peak ratio — reuses FFT from PSD (zero redundancy)
        acf = torch.fft.ifft(psd).real
        zero_lag = acf[0]
        if zero_lag.item() < eps:
            caf_ratio = torch.tensor(0.0, device=iq.device)
        else:
            acf_normed = torch.abs(acf / zero_lag)
            caf_ratio  = acf_normed[1:N // 4].max()

        result = torch.stack([
            mean_pow, papr, kurt, skew,
            flatness, entropy_norm, bw_norm, caf_ratio
        ])

        result = torch.where(torch.isfinite(result), result,
                             torch.zeros_like(result))
        return result


    def preprocess(self, iq_np: np.ndarray) -> dict:
        """
        Full preprocessing: IQ numpy -> dict of numpy arrays ready for TRT.
        One CPU -> GPU transfer, all computation on GPU, numpy arrays out.
        """
        iq = torch.from_numpy(iq_np).to(self.device)

        spec_np = self.iq_to_spectrogram(iq)
        result  = {"spectrogram": spec_np}

        if self.feat_norm is not None:
            feat = self.compute_features(iq)
            feat = self.feat_norm.transform_gpu(feat)
            result["features"] = feat.unsqueeze(0).cpu().numpy().astype(np.float32)

        return result


# Argument parsing
def parse_args():
    p = argparse.ArgumentParser(
        description="Real-time GNSS interference classification — Jetson Nano demo")
    p.add_argument("--trt",      required=True)
    p.add_argument("--uri",      default="ip:192.168.2.1")
    p.add_argument("--freq",     type=int, default=GPS_L1_HZ)
    p.add_argument("--gain-mode",default="slow_attack",
                   choices=["slow_attack", "fast_attack", "manual", "hybrid"])
    p.add_argument("--gain",     type=float, default=40.0)
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--snap",     default=None)
    p.add_argument("--csv-out",  default="demo_log.csv")
    p.add_argument("--warmup",   type=int, default=5)
    p.add_argument("--feat-norm", default=None)
    p.add_argument("--fifo",     default=None,
                   help="Path to named FIFO for IQ tee to GNSS-SDR "
                        "(e.g. /tmp/gnss_iq_fifo)")
    return p.parse_args()


# TensorRT engine
class TRTClassifier:
    def __init__(self, trt_path: str):

        # Local imports
        import tensorrt as trt
        import pycuda.driver as cuda

        # DO NOT import pycuda.autoinit — it creates a separate CUDA context
        # that conflicts with PyTorch's. Instead, attach to the primary context
        # that PyTorch already initialized.

        cuda.init()
        self._cuda_ctx = cuda.Device(0).retain_primary_context()
        self._cuda_ctx.push()

        self._cuda = cuda
        self._trt  = trt

        print(f"[TRT] Loading engine from {trt_path} ...")
        logger  = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(trt_path, "rb") as f:
            engine_bytes = f.read()

        self.engine  = runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        n = self.engine.num_bindings
        print(f"[TRT] Engine loaded. Bindings: {n}")

        self._bindings = [None] * n
        self.inputs  = {}
        self.outputs = {}

        for i in range(n):
            name  = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = self.engine.get_binding_dtype(i)
            kind  = "INPUT" if self.engine.binding_is_input(i) else "OUTPUT"
            print(f"[TRT]   [{kind}] {name}  shape={shape}  dtype={dtype}")

            n_elems = int(np.prod(shape))
            h_buf   = cuda.pagelocked_empty(n_elems, dtype=np.float32)
            d_buf   = cuda.mem_alloc(h_buf.nbytes)
            self._bindings[i] = int(d_buf)

            entry = dict(idx=i, h_buf=h_buf, d_buf=d_buf, shape=shape)
            if self.engine.binding_is_input(i):
                self.inputs[name]  = entry
            else:
                self.outputs[name] = entry

        self._stream = cuda.Stream()
        self.is_fusion = len(self.inputs) > 1
        print(f"[TRT] Mode: {'fusion' if self.is_fusion else 'spectrogram-only'}")
        print()


    def _copy_inputs(self, input_dict):
        for name, entry in self.inputs.items():
            arr = input_dict.get(name)
            if arr is None:
                raise ValueError(f"[TRT] Missing input '{name}'.")
            np.copyto(entry["h_buf"], arr.ravel())
            self._cuda.memcpy_htod_async(entry["d_buf"], entry["h_buf"], self._stream)


    def _read_outputs(self):
        for entry in self.outputs.values():
            self._cuda.memcpy_dtoh_async(entry["h_buf"], entry["d_buf"], self._stream)
        self._stream.synchronize()
        first = next(iter(self.outputs.values()))
        return first["h_buf"].copy().ravel()


    def infer(self, input_dict):
        self._copy_inputs(input_dict)
        torch.cuda.synchronize()  # ensure PyTorch kernels are done
        self.context.execute_async_v2(self._bindings, self._stream.handle)
        return self._read_outputs()


    def infer_timed(self, input_dict):
        self._copy_inputs(input_dict)
        torch.cuda.synchronize()  # ensure PyTorch kernels are done
        t0 = time.perf_counter()
        self.context.execute_async_v2(self._bindings, self._stream.handle)
        self._stream.synchronize()
        gpu_ms = (time.perf_counter() - t0) * 1e3
        for entry in self.outputs.values():
            self._cuda.memcpy_dtoh_async(entry["h_buf"], entry["d_buf"], self._stream)
        self._stream.synchronize()
        first = next(iter(self.outputs.values()))
        return first["h_buf"].copy().ravel(), gpu_ms


# Softmax helper
def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# IIO source (PlutoSDR via libiio)
class IIOSource:
    def __init__(self, uri, freq, samp_rate, bandwidth, gain_mode, gain_db,
                 tee=None):
        self._tee = tee
        try:
            import iio
        except ImportError:
            print("[FAIL] pylibiio not installed.  pip3 install pylibiio")
            sys.exit(1)

        print(f"[IIO] Opening context at {uri} ...")
        try:
            self._ctx = iio.Context(uri)
        except Exception as e:
            print(f"[FAIL] Cannot open IIO context: {e}")
            sys.exit(1)

        phy = self._ctx.find_device("ad9361-phy")
        rx_ch = phy.find_channel("voltage0", False)
        lo_ch = phy.find_channel("altvoltage0", True)
        rx_ch.attrs["rf_bandwidth"].value = str(bandwidth)
        rx_ch.attrs["sampling_frequency"].value = str(samp_rate)
        rx_ch.attrs["gain_control_mode"].value = gain_mode
        if gain_mode == "manual":
            rx_ch.attrs["hardwaregain"].value = str(max(0.0, min(71.0, gain_db)))
        lo_ch.attrs["frequency"].value = str(freq)

        rx_dev = self._ctx.find_device("cf-ad9361-lpc")
        self._i = rx_dev.find_channel("voltage0", False)
        self._q = rx_dev.find_channel("voltage1", False)
        self._i.enabled = True
        self._q.enabled = True
        self._buf = iio.Buffer(rx_dev, IIO_BUFFER_SIZE, False)

        print(f"[IIO] Configured — {freq/1e6:.3f} MHz  {samp_rate/1e6:.1f} MSPS  {gain_mode}")
        print("[IIO] Settling 1 s ...")
        time.sleep(1.0)
        print()

    def read_window(self):
        chunks, total = [], 0
        while total < WINDOW_SAMPLES:
            self._buf.refill()
            raw = self._buf.read()
            s = np.frombuffer(raw, dtype=np.int16)

            # Tee raw int16 IQ to GNSS-SDR FIFO (non-blocking, per-refill
            # so GNSS-SDR gets a continuous stream, not 20 ms bursts)
            if self._tee is not None:
                self._tee.write(s)

            iq  = (s[0::2].astype(np.float32) + 1j * s[1::2].astype(np.float32)) / 2048.0
            need = WINDOW_SAMPLES - total
            chunks.append(iq[:need])
            total += len(chunks[-1])
        return np.concatenate(chunks)


class SnapshotSource:
    def __init__(self, path):
        self._iq  = np.load(path).astype(np.complex64)
        self._pos = 0
        n_win = len(self._iq) // WINDOW_SAMPLES
        print(f"[SNAP] Loaded {path}:  {len(self._iq):,} samples  ({n_win} windows)")
        if n_win == 0:
            print(f"[FAIL] Snapshot too short.")
            sys.exit(1)
        print()


    def read_window(self):
        end = self._pos + WINDOW_SAMPLES
        if end > len(self._iq):
            self._pos, end = 0, WINDOW_SAMPLES
        win = self._iq[self._pos:end]
        self._pos = end
        return win


# Console display
def _bar(conf, width=20):
    return "█" * int(round(conf * width)) + "░" * (width - int(round(conf * width)))

def print_header():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   GNSS Interference Classifier — Real-Time Demo              ║")
    print("║   Jetson Nano  ·  TensorRT  ·  GPU Preprocessing             ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Win │ Prediction              │ Conf  │ Bar          │  ms  ║")
    print("╠══════════════════════════════════════════════════════════════╣")

def print_row(win_idx, label, conf, infer_ms, pipeline_ms):
    short = SHORT_LABEL.get(label, label[:19].ljust(19))
    print(f"║ {win_idx:4d} │ {short}│ {conf:4.1%} │ {_bar(conf, 14)} │{pipeline_ms:4.1f} ║",
          flush=True)

def print_footer():
    print("╚══════════════════════════════════════════════════════════════╝")


# Latency summary
def print_latency_summary(infer_times, preproc_times, pipeline_times, n_windows):
    print()
    print("=" * 62)
    print("  Latency Summary")
    print("=" * 62)
    arr_i = np.array(infer_times)
    arr_r = np.array(preproc_times)
    arr_p = np.array(pipeline_times)
    print(f"  Windows processed     : {n_windows}")
    print(f"  Window duration       : {WINDOW_SEC*1000:.0f} ms  "
          f"({WINDOW_SAMPLES:,} samples @ {SAMPLE_RATE/1e6:.0f} MSPS)")
    print()
    print(f"  GPU preprocessing (STFT + features + norm):")
    print(f"    Mean   : {arr_r.mean():.2f} ms")
    print(f"    Std    : {arr_r.std():.2f} ms")
    print(f"    Min    : {arr_r.min():.2f} ms")
    print(f"    Max    : {arr_r.max():.2f} ms")
    print(f"    P95    : {np.percentile(arr_r, 95):.2f} ms")
    print()
    print(f"  TRT inference only (GPU execution):")
    print(f"    Mean   : {arr_i.mean():.2f} ms")
    print(f"    Std    : {arr_i.std():.2f} ms")
    print(f"    Min    : {arr_i.min():.2f} ms")
    print(f"    Max    : {arr_i.max():.2f} ms")
    print(f"    P95    : {np.percentile(arr_i, 95):.2f} ms")
    print()
    print(f"  Full pipeline (preproc + H2D + infer + D2H):")
    print(f"    Mean   : {arr_p.mean():.2f} ms")
    print(f"    Std    : {arr_p.std():.2f} ms")
    print(f"    Min    : {arr_p.min():.2f} ms")
    print(f"    Max    : {arr_p.max():.2f} ms")
    print(f"    P95    : {np.percentile(arr_p, 95):.2f} ms")
    print()
    budget_ok = arr_p.mean() < WINDOW_SEC * 1000
    print(f"  Real-time budget ({WINDOW_SEC*1000:.0f} ms): "
          f"{'✓ met' if budget_ok else '✗ EXCEEDED'} "
          f"(mean pipeline {arr_p.mean():.1f} ms)")
    print()
    print(f"  Speedup vs previous CPU preprocessing:")
    print(f"    Previous: ~177 ms  →  Current: {arr_p.mean():.1f} ms  "
          f"({177.0 / max(arr_p.mean(), 0.1):.1f}x)")
    print("=" * 62)


# Main inference loop
def run(args):
    # IQ tee for parallel GNSS-SDR
    tee = None
    if args.fifo:
        try:
            tee = IQTee(args.fifo)
        except (FileNotFoundError, ValueError) as e:
            print(f"[WARN] IQ tee disabled: {e}")
            tee = None

    if args.snap:
        source = SnapshotSource(args.snap)
    else:
        source = IIOSource(args.uri, args.freq, SAMPLE_RATE,
                           RF_BANDWIDTH_HZ, args.gain_mode, args.gain,
                           tee=tee)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize PyTorch's CUDA context BEFORE TRTClassifier,
    # so PyCUDA can attach to the same primary context.
    if device.type == "cuda":
        torch.cuda.init()
        _ = torch.zeros(1, device=device)  # force context creation

    clf = TRTClassifier(args.trt)

    feat_norm = None
    if clf.is_fusion:
        feat_norm_path = args.feat_norm
        if feat_norm_path is None:
            candidates = list(Path(".").rglob("feature_norm_stats.json"))
            if candidates:
                feat_norm_path = str(candidates[0])
                print(f"[INFO] Auto-detected feat norm: {feat_norm_path}")
            else:
                print("[FAIL] Fusion model needs --feat-norm")
                sys.exit(1)
        feat_norm = FeatureNormalizer.load(feat_norm_path)
        print(f"[INFO] Feature normalizer loaded from {feat_norm_path}")
        print()

    gpu_pre = GPUPreprocessor(device, feat_norm if clf.is_fusion else None)

    csv_path = Path(args.csv_out)
    csv_file = csv_path.open("w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "window", "timestamp_s", "predicted_class", "confidence",
        *[f"p_{c}" for c in CLASS_NAMES],
        "preproc_gpu_ms", "infer_gpu_ms", "pipeline_ms",
    ])

    # Warmup both PyTorch CUDA kernels and TRT engine
    print(f"[INFO] Warming up GPU pipeline ({args.warmup} iterations) ...")
    dummy_iq = np.zeros(WINDOW_SAMPLES, dtype=np.complex64)
    for _ in range(args.warmup):
        d = gpu_pre.preprocess(dummy_iq)
        clf.infer(d)
    print("[INFO] Warmup complete.")
    print()

    infer_times, preproc_times, pipeline_times = [], [], []
    window_idx = 0
    t_start = time.perf_counter()

    print_header()

    try:
        while True:
            if args.duration > 0 and time.perf_counter() - t_start >= args.duration:
                break

            iq = source.read_window()

            t_pipe = time.perf_counter()

            t_pre = time.perf_counter()
            input_dict = gpu_pre.preprocess(iq)
            preproc_ms = (time.perf_counter() - t_pre) * 1e3

            logits, infer_ms = clf.infer_timed(input_dict)
            pipeline_ms = (time.perf_counter() - t_pipe) * 1e3

            probs      = softmax(logits)
            pred_idx   = int(np.argmax(probs))
            pred_label = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx])

            if window_idx >= args.warmup:
                infer_times.append(infer_ms)
                preproc_times.append(preproc_ms)
                pipeline_times.append(pipeline_ms)

            print_row(window_idx, pred_label, confidence, infer_ms, pipeline_ms)

            writer.writerow([
                window_idx,
                f"{time.perf_counter() - t_start:.3f}",
                pred_label, f"{confidence:.6f}",
                *[f"{float(p):.6f}" for p in probs],
                f"{preproc_ms:.3f}", f"{infer_ms:.3f}", f"{pipeline_ms:.3f}",
            ])
            csv_file.flush()
            window_idx += 1

    except KeyboardInterrupt:
        pass
    finally:
        print_footer()
        if tee is not None:
            tee.close()
        csv_file.close()

    if infer_times:
        print_latency_summary(infer_times, preproc_times, pipeline_times, window_idx)
    print(f"\n[INFO] Log saved → {args.csv_out}")

    # Pop the PyCUDA context to prevent abort on module cleanup
    try:
        clf._cuda_ctx.pop()
    except Exception:
        pass


def main():
    args = parse_args()
    if not Path(args.trt).exists():
        print(f"[FAIL] TRT engine not found: {args.trt}")
        sys.exit(1)
    if args.snap and not Path(args.snap).exists():
        print(f"[FAIL] Snapshot not found: {args.snap}")
        sys.exit(1)
    run(args)


if __name__ == "__main__":
    main()
