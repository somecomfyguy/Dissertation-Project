"""
export_onnx.py
==============
Export the trained FusionModel (custom_cnn backbone) to ONNX.

Run on the LAPTOP (not Jetson) -- ONNX is platform-independent.

Usage
-----
    python export_onnx.py \
        --checkpoint output/results_11classes_regularized_fusion/fusion_custom_cnn/best_model.pth \
        --output fusion_custom_cnn.onnx \
        [--opset 11]

TRT 8.2 compatibility — fully 4D approach
------------------------------------------
TRT 8.2.1.8 cannot reliably transition between 4D and 2D tensors: every
approach (Flatten, ReduceMean+Gemm, AdaptiveAvgPool2d+Conv2d, Squeeze,
Reshape) fails due to shape analysis bugs in channel tracking or axis
resolution.

The solution: NEVER leave 4D. The entire network operates on 4D tensors:

  CNN branch:   Conv blocks -> AvgPool2d(8) -> (1, 256, 1, 1)
  Feat branch:  Unsqueeze (1,8) -> (1,8,1,1) -> Conv2d(1x1) layers -> (1,32,1,1)
  Concat:       (1, 256, 1, 1) + (1, 32, 1, 1) -> (1, 288, 1, 1)  [same ndims]
  Classify:     Conv2d(288, 11, 1) -> (1, 11, 1, 1)

Linear layers become equivalent 1x1 Conv2d. Concat operates on same-dim
4D tensors. No Flatten, no Reshape, no Gemm, no ReduceMean.

Output shape: (1, 11, 1, 1) — squeezed in the inference wrapper.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

from modules.nn_module.fusion_model import build_fusion_model
from modules.common.types import N_FEATURES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES     = 11
BACKBONE        = "custom_cnn"
BACKBONE_DIM    = 256
FEAT_EMBED      = 32
SPECTROGRAM_H   = 128
SPECTROGRAM_W   = 128
SPECTROGRAM_C   = 3
# After 4x MaxPool(2) on 128x128 input, spatial dims are 8x8
FINAL_SPATIAL   = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_to_conv1x1(linear):
    """Convert a Linear(in, out) to an equivalent Conv2d(in, out, 1)."""
    conv = nn.Conv2d(linear.in_features, linear.out_features,
                     kernel_size=1, bias=(linear.bias is not None))
    conv.weight = nn.Parameter(
        linear.weight.data.reshape(
            linear.out_features, linear.in_features, 1, 1
        ).clone()
    )
    if linear.bias is not None:
        conv.bias = nn.Parameter(linear.bias.data.clone())
    return conv


def _fold_bn1d(linear, bn):
    """Absorb BatchNorm1d into its preceding Linear (eval mode only)."""
    scale = bn.weight.data / torch.sqrt(bn.running_var + bn.eps)
    W_new = linear.weight.data * scale.unsqueeze(1)
    b_new = (linear.bias.data - bn.running_mean) * scale + bn.bias.data
    folded = nn.Linear(linear.in_features, linear.out_features, bias=True)
    folded.weight = nn.Parameter(W_new)
    folded.bias   = nn.Parameter(b_new)
    return folded


# ---------------------------------------------------------------------------
# TRT-compatible fully-4D export model
# ---------------------------------------------------------------------------

class TRTExportModel(nn.Module):
    """
    Fully-4D wrapper for TRT 8.2. No dimensional transitions anywhere.

    Operations used (all TRT 8.2-native on 4D tensors):
        Conv, ReLU, MaxPool, AveragePool, Unsqueeze, Concat, Add
    """

    def __init__(self, backbone, feat_conv_layers, classifier_conv):
        super().__init__()
        self.backbone = backbone        # conv blocks + AvgPool2d(8) -> (1,256,1,1)
        self.feat_conv = feat_conv_layers  # Conv2d(1x1) chain -> (1,32,1,1)
        self.classifier = classifier_conv  # Conv2d(288,11,1) -> (1,11,1,1)

    def forward(self, spectrogram, features):
        # CNN branch — fully 4D
        cnn_out = self.backbone(spectrogram)          # (1, 256, 1, 1)

        # Feature branch — unsqueeze to 4D, then 1x1 Conv2d layers
        feat_4d = features.unsqueeze(2).unsqueeze(3)  # (1, 8) -> (1, 8, 1, 1)
        feat_out = self.feat_conv(feat_4d)             # (1, 32, 1, 1)

        # Concat — both 4D with identical spatial dims (1x1)
        combined = torch.cat([cnn_out, feat_out], dim=1)  # (1, 288, 1, 1)

        # Classify — 1x1 Conv2d
        return self.classifier(combined)               # (1, 11, 1, 1)


# ---------------------------------------------------------------------------
# Model loading and restructuring
# ---------------------------------------------------------------------------

def load_model(checkpoint_path):
    """Build, load, and restructure the model for fully-4D TRT export."""
    print(f"[export] Building FusionModel ({BACKBONE}, {NUM_CLASSES} classes)...")
    model = build_fusion_model(
        backbone_name=BACKBONE,
        num_classes=NUM_CLASSES,
        freeze_backbone=False,
        dropout=0.4,
    )

    print(f"[export] Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    print("[export] Model in eval mode.")

    # -- 1. Fold BatchNorm1d into FeatureMLP Linear layers ---------------
    layers = list(model.feat_mlp.net)
    folded_layers = [
        _fold_bn1d(layers[0], layers[1]),   # Linear(8,64) + BN1d(64)
        layers[2],                          # ReLU
        _fold_bn1d(layers[3], layers[4]),   # Linear(64,32) + BN1d(32)
        layers[5],                          # ReLU
    ]
    print("[export] Folded BatchNorm1d into Linear weights.")

    # -- 2. Convert FeatureMLP Linear layers to 1x1 Conv2d ---------------
    feat_conv = nn.Sequential(
        _linear_to_conv1x1(folded_layers[0]),  # Conv2d(8, 64, 1)
        folded_layers[1],                      # ReLU
        _linear_to_conv1x1(folded_layers[2]),  # Conv2d(64, 32, 1)
        folded_layers[3],                      # ReLU
    )
    print("[export] Converted FeatureMLP to 1x1 Conv2d layers.")

    # -- 3. Replace backbone classifier with AvgPool2d(8) ----------------
    # Using explicit AvgPool2d (not Adaptive) -- simpler ONNX node, no
    # dynamic kernel computation that might confuse TRT's shape analyzer.
    model.backbone.classifier = nn.AvgPool2d(kernel_size=FINAL_SPATIAL)
    print(f"[export] Backbone classifier -> AvgPool2d({FINAL_SPATIAL}).")

    # -- 4. Convert final classifier Linear to 1x1 Conv2d ----------------
    final_linear = model.classifier[1]  # Linear(288, 11)
    W = final_linear.weight.data        # (11, 288)
    b = final_linear.bias.data          # (11,)
    classifier_conv = nn.Conv2d(
        BACKBONE_DIM + FEAT_EMBED, NUM_CLASSES, kernel_size=1, bias=True
    )
    classifier_conv.weight = nn.Parameter(
        W.reshape(NUM_CLASSES, BACKBONE_DIM + FEAT_EMBED, 1, 1).clone()
    )
    classifier_conv.bias = nn.Parameter(b.clone())
    print("[export] Final classifier -> Conv2d(288, 11, 1).")

    # -- 5. Build export wrapper -----------------------------------------
    export_model = TRTExportModel(
        backbone=model.backbone,
        feat_conv_layers=feat_conv,
        classifier_conv=classifier_conv,
    )
    export_model.eval()

    # -- 6. Sanity check: numerical equivalence --------------------------
    with torch.no_grad():
        test_spec = torch.randn(1, SPECTROGRAM_C, SPECTROGRAM_H, SPECTROGRAM_W)
        test_feat = torch.randn(1, N_FEATURES)

        # Original forward (manual, using modified components)
        cnn_pooled = model.backbone(test_spec)   # (1, 256, 1, 1) via AvgPool2d
        cnn_flat = cnn_pooled.flatten(1)          # (1, 256)

        feat_out_orig = nn.Sequential(*folded_layers)(test_feat)  # (1, 32)
        combined = torch.cat([cnn_flat, feat_out_orig], dim=1)    # (1, 288)
        original_logits = final_linear(combined)                   # (1, 11)

        # Wrapper forward
        wrapper_logits = export_model(test_spec, test_feat)        # (1, 11, 1, 1)
        wrapper_flat = wrapper_logits.flatten(1)                   # (1, 11)

        max_diff = (original_logits - wrapper_flat).abs().max().item()
        print(f"[export] Sanity check: max diff = {max_diff:.2e}")
        assert max_diff < 1e-4, f"Mismatch: {max_diff:.2e}"

    print("[export] TRTExportModel ready (fully 4D).")
    return export_model


# ---------------------------------------------------------------------------
# ONNX export and verification
# ---------------------------------------------------------------------------

def export(model, output_path, opset):
    dummy_spec = torch.zeros(1, SPECTROGRAM_C, SPECTROGRAM_H, SPECTROGRAM_W)
    dummy_feat = torch.zeros(1, N_FEATURES)

    print(f"[export] Exporting to ONNX (opset {opset})...")
    torch.onnx.export(
        model,
        args=(dummy_spec, dummy_feat),
        f=output_path,
        opset_version=opset,
        input_names=["spectrogram", "features"],
        output_names=["logits"],
        do_constant_folding=True,
        verbose=False,
    )
    print(f"[export] Saved: {output_path}")


def verify(output_path):
    try:
        import onnx
    except ImportError:
        print("[verify] 'onnx' not installed -- skipping.")
        return

    print("[verify] Running onnx.checker...")
    model_proto = onnx.load(output_path)
    onnx.checker.check_model(model_proto)
    print("[verify] Passed.")

    for inp in model_proto.graph.input:
        shape = [d.dim_param if d.dim_param else d.dim_value
                 for d in inp.type.tensor_type.shape.dim]
        print(f"         input  '{inp.name}': {shape}")
    for out in model_proto.graph.output:
        shape = [d.dim_param if d.dim_param else d.dim_value
                 for d in out.type.tensor_type.shape.dim]
        print(f"         output '{out.name}': {shape}")

    print(f"[verify] Graph nodes ({len(model_proto.graph.node)}):")
    for i, node in enumerate(model_proto.graph.node):
        print(f"         [{i:3d}] op={node.op_type:<20s} name={node.name}")

    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("[verify] 'onnxruntime' not installed -- skipping inference.")
        return

    print("[verify] Running onnxruntime inference check...")
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    dummy_spec = np.zeros((1, SPECTROGRAM_C, SPECTROGRAM_H, SPECTROGRAM_W),
                          dtype=np.float32)
    dummy_feat = np.zeros((1, N_FEATURES), dtype=np.float32)
    outputs = sess.run(["logits"],
                       {"spectrogram": dummy_spec, "features": dummy_feat})
    logits = outputs[0]
    expected = (1, NUM_CLASSES, 1, 1)
    print(f"[verify] Output shape: {logits.shape}  (expected: {expected})")
    assert logits.shape == expected, f"Shape mismatch: {logits.shape}"
    print("[verify] All checks passed.")


def main():
    parser = argparse.ArgumentParser(
        description="Export FusionModel to ONNX (fully-4D for TRT 8.2)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="fusion_custom_cnn.onnx")
    parser.add_argument("--opset", type=int, default=11, choices=[11, 12, 13],
                        help="ONNX opset (default: 11 -- safest for TRT 8.2)")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"[error] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    model = load_model(args.checkpoint)
    export(model, args.output, args.opset)

    if not args.no_verify:
        verify(args.output)

    try:
        import onnx
        print("[export] Consolidating into single ONNX file...")
        m = onnx.load(args.output)
        onnx.save(m, args.output)
    except ImportError:
        pass

    print(f"\n[done] Copy {args.output} to the Jetson and run convert_trt.py.")
    print("       Output shape is (1, 11, 1, 1).")


if __name__ == "__main__":
    main()