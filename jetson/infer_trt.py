"""
infer_trt.py
============
TensorRT inference wrapper for the Jetson Nano.

Output shape: (1, 11, 1, 1) -- squeezed to (1, 11) internally.

Usage:  python3 infer_trt.py --engine fusion_custom_cnn.trt [--benchmark]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


CLASS_NAMES = [
    "clean", "jam_chirp", "jam_cw", "jam_multitone", "jam_pulse",
    "jam_wideband", "spoof_dynamic", "spoof_matched_static",
    "spoof_meaconing", "spoof_overpowered", "spoof_seamless",
]

SPEC_C, SPEC_H, SPEC_W = 3, 128, 128
N_FEATURES  = 8
NUM_CLASSES = len(CLASS_NAMES)
OUT_SHAPE   = (1, NUM_CLASSES, 1, 1)   # fully-4D export output


class TRTInferenceEngine:
    def __init__(self, engine_path):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa

        self._trt  = trt
        self._cuda = cuda

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        print(f"[TRT] Loading engine: {engine_path}")
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to load: {engine_path}")
        self._context = self._engine.create_execution_context()

        self._idx_spec   = self._engine.get_binding_index("spectrogram")
        self._idx_feat   = self._engine.get_binding_index("features")
        self._idx_logits = self._engine.get_binding_index("logits")

        self._h_spec   = cuda.pagelocked_empty((1, SPEC_C, SPEC_H, SPEC_W), np.float32)
        self._h_feat   = cuda.pagelocked_empty((1, N_FEATURES), np.float32)
        self._h_logits = cuda.pagelocked_empty(OUT_SHAPE, np.float32)

        self._d_spec   = cuda.mem_alloc(self._h_spec.nbytes)
        self._d_feat   = cuda.mem_alloc(self._h_feat.nbytes)
        self._d_logits = cuda.mem_alloc(self._h_logits.nbytes)

        n = self._engine.num_bindings
        self._bindings = [0] * n
        self._bindings[self._idx_spec]   = int(self._d_spec)
        self._bindings[self._idx_feat]   = int(self._d_feat)
        self._bindings[self._idx_logits] = int(self._d_logits)
        self._stream = cuda.Stream()
        print(f"[TRT] Ready ({n} bindings).")

    def predict(self, spectrogram, features):
        cuda = self._cuda
        spec = np.asarray(spectrogram, dtype=np.float32)
        if spec.ndim == 2:
            spec = np.stack([spec]*3, axis=0)[np.newaxis]
        elif spec.ndim == 3:
            spec = spec[np.newaxis]
        feat = np.asarray(features, dtype=np.float32).reshape(1, N_FEATURES)

        np.copyto(self._h_spec, spec)
        np.copyto(self._h_feat, feat)
        cuda.memcpy_htod_async(self._d_spec, self._h_spec, self._stream)
        cuda.memcpy_htod_async(self._d_feat, self._h_feat, self._stream)
        self._context.execute_async_v2(self._bindings, self._stream.handle)
        cuda.memcpy_dtoh_async(self._h_logits, self._d_logits, self._stream)
        self._stream.synchronize()

        logits = self._h_logits.copy().reshape(1, NUM_CLASSES)
        probs  = _softmax(logits[0])
        idx    = int(np.argmax(probs))
        return CLASS_NAMES[idx], float(probs[idx]), logits

    def benchmark(self, n_runs=100):
        d = np.zeros((SPEC_H, SPEC_W), np.float32)
        f = np.zeros(N_FEATURES, np.float32)
        for _ in range(5):
            self.predict(d, f)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.predict(d, f)
            times.append((time.perf_counter() - t0) * 1000)
        m, s = float(np.mean(times)), float(np.std(times))
        print(f"[bench] {n_runs} runs: {m:.2f} +/- {s:.2f} ms")
        return m

    def destroy(self):
        self._d_spec.free(); self._d_feat.free(); self._d_logits.free()

    def __enter__(self):  return self
    def __exit__(self, *_): self.destroy()


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True)
    p.add_argument("--spectrogram", default=None)
    p.add_argument("--features", default=None)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--n-runs", type=int, default=100)
    a = p.parse_args()

    with TRTInferenceEngine(a.engine) as eng:
        if a.benchmark:
            eng.benchmark(a.n_runs)
        spec = np.load(a.spectrogram).astype(np.float32) if a.spectrogram else np.zeros((SPEC_H,SPEC_W), np.float32)
        feat = np.load(a.features).astype(np.float32)    if a.features    else np.zeros(N_FEATURES, np.float32)
        t0 = time.perf_counter()
        label, conf, logits = eng.predict(spec, feat)
        ms = (time.perf_counter() - t0) * 1000
        print(f"\n  Predicted : {label}\n  Confidence: {conf:.1%}\n  Latency   : {ms:.2f} ms")
        probs = _softmax(logits[0])
        for i in np.argsort(probs)[::-1][:3]:
            print(f"    {CLASS_NAMES[i]:<28s}  {probs[i]:.1%}")


if __name__ == "__main__":
    main()