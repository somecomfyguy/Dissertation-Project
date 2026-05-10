"""
convert_trt.py
==============
Convert the exported ONNX model to a TensorRT engine on the Jetson Nano.

MUST run on the Jetson Nano — TRT engines are device- and driver-specific.
A TRT engine built on the laptop will NOT run on the Nano.

Environment (Jetson Nano, JetPack 4.6.x)
-----------------------------------------
    TensorRT  : 8.2.1.8
    cuDNN     : 8.2.1.32
    Python TRT: import tensorrt as trt  (ships with JetPack)
    pycuda    : install if missing:  pip3 install pycuda --user

Usage
-----
    python3 convert_trt.py \
        --onnx fusion_custom_cnn.onnx \
        --output fusion_custom_cnn.trt \
        [--fp16] [--workspace 512]

TRT 8.2 API notes
-----------------
- EXPLICIT_BATCH flag is required for ONNX models with a batch dimension.
- build_serialized_network() replaces the deprecated build_engine() in TRT 8.x.
- max_workspace_size is set via config (not builder) in TRT 8.x.
- Dynamic shape profiles: even though ONNX was exported with dynamic batch,
  we lock batch=1 here (min=opt=max) for deterministic Jetson inference.
  This lets TRT apply more aggressive kernel fusion.

FP16 note
---------
The Jetson Nano Maxwell GPU supports FP16 (half precision). Enabling it
roughly halves memory bandwidth for the convolutional layers and typically
reduces latency 20-40% with negligible accuracy loss on INT8-quantized
feature inputs. FP16 is recommended and enabled by default.
"""

import argparse
import sys
from pathlib import Path


def build_engine(onnx_path: str, output_path: str,
                 fp16: bool = True, workspace_mb: int = 512) -> None:
    """
    Parse the ONNX model and build a TensorRT engine, saving it to disk.

    Args:
        onnx_path:    Path to the .onnx file.
        output_path:  Destination path for the serialized .trt engine.
        fp16:         Enable FP16 precision (recommended for Jetson Nano).
        workspace_mb: GPU workspace limit in MiB. 512 MB is safe for the
                      Nano's 4 GB shared memory; reduce if OOM.
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("[error] 'tensorrt' module not found.")
        print("        This script must run on the Jetson Nano (JetPack 4.6).")
        sys.exit(1)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    print(f"[convert] TensorRT version : {trt.__version__}")
    print(f"[convert] ONNX source      : {onnx_path}")
    print(f"[convert] FP16 enabled     : {fp16}")
    print(f"[convert] Workspace        : {workspace_mb} MiB")

    # ── Builder + Network ──────────────────────────────────────────────────
    # EXPLICIT_BATCH: required for models with a dynamic batch dimension.
    # Without this flag, TRT treats the first dim as implicit batch, which
    # breaks ONNX models that include batch in the graph.
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser  = trt.OnnxParser(network, TRT_LOGGER)

    print(f"[convert] Parsing ONNX graph...")
    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())

    if not ok:
        print("[error] ONNX parsing failed:")
        for i in range(parser.num_errors):
            err = parser.get_error(i)
            print(f"         [{i}] {err}")
        sys.exit(1)

    # Report what TRT sees as inputs after parsing
    print(f"[convert] Network inputs ({network.num_inputs}):")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"          [{i}] '{inp.name}' shape={inp.shape} "
              f"dtype={inp.dtype}")
    print(f"[convert] Network outputs ({network.num_outputs}):")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"          [{i}] '{out.name}' shape={out.shape}")

    # ── Builder Config ─────────────────────────────────────────────────────
    config = builder.create_builder_config()

    # TRT 8.2 uses max_workspace_size in bytes (deprecated in TRT 9+)
    config.max_workspace_size = workspace_mb * (1 << 20)

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[convert] FP16 flag set.")
        else:
            print("[convert] WARNING: platform does not report fast FP16. "
                  "Falling back to FP32.")

    # ── Optimization Profile ───────────────────────────────────────────────
    # Even though the ONNX has a dynamic batch axis, we constrain TRT to
    # batch=1 (min=opt=max). This allows maximum kernel fusion for the
    # single-sample inference path used in the embedded demonstrator.
    profile = builder.create_optimization_profile()

    # Input name → (min_shape, opt_shape, max_shape)
    # These must match the input_names used in export_onnx.py exactly.
    profile.set_shape("spectrogram",
                      min=(1, 3, 128, 128),
                      opt=(1, 3, 128, 128),
                      max=(1, 3, 128, 128))
    profile.set_shape("features",
                      min=(1, 8),
                      opt=(1, 8),
                      max=(1, 8))

    config.add_optimization_profile(profile)
    print("[convert] Optimization profile set (batch=1, fixed).")

    # ── Build & Serialize ──────────────────────────────────────────────────
    print("[convert] Building TRT engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("[error] Engine build failed. Check TRT warnings above.")
        sys.exit(1)

    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    size_mb = Path(output_path).stat().st_size / (1 << 20)
    print(f"[convert] Engine saved: {output_path}  ({size_mb:.1f} MiB)")
    print("[convert] Done. ✓")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX model to TensorRT engine on Jetson Nano"
    )
    parser.add_argument(
        "--onnx", required=True,
        help="Path to the exported .onnx file",
    )
    parser.add_argument(
        "--output", default="fusion_custom_cnn.trt",
        help="Output .trt engine path (default: fusion_custom_cnn.trt)",
    )
    parser.add_argument(
        "--fp16", action="store_true", default=True,
        help="Enable FP16 precision (default: True)",
    )
    parser.add_argument(
        "--no-fp16", dest="fp16", action="store_false",
        help="Disable FP16 (force FP32)",
    )
    parser.add_argument(
        "--workspace", type=int, default=512,
        help="GPU workspace in MiB (default: 512)",
    )
    args = parser.parse_args()

    if not Path(args.onnx).exists():
        print(f"[error] ONNX file not found: {args.onnx}")
        sys.exit(1)

    build_engine(args.onnx, args.output, fp16=args.fp16,
                 workspace_mb=args.workspace)

    print("\nAlternative: trtexec one-liner (produces identical engine):")
    print(f"  trtexec --onnx={args.onnx} \\")
    print(f"          --saveEngine={args.output} \\")
    print(f"          --fp16 \\")
    print(f"          --workspace={args.workspace} \\")
    print(f"          --minShapes=spectrogram:1x3x128x128,features:1x8 \\")
    print(f"          --optShapes=spectrogram:1x3x128x128,features:1x8 \\")
    print(f"          --maxShapes=spectrogram:1x3x128x128,features:1x8")


if __name__ == "__main__":
    main()
