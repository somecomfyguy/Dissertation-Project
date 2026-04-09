"""
Custom lightweight CNN for GNSS interference classification, designed for deployment 
on the Jetson Nano (Maxwell GPU, 4GB shared memory) with a target of <500K parameters 
and <5ms inference at 128x128 input.

Architecture principles drawn from the literature:
    - Depthwise separable convolutions reduce parameters and FLOPs by
      approximately 8-9x per layer compared to standard convolutions
      (Howard et al., 2017 — MobileNets; validated for GNSS interference
      by Elango et al., 2022, ICL-GNSS).
    - Asymmetric/factored convolution blocks reduce parameters further
      while preserving spatial feature extraction capability
      (Jiang et al., 2024/2025 — ACSNet, 50% parameter reduction).
    - Global Average Pooling eliminates the dense flattening layer that
      dominates parameter count in traditional CNNs, following the
      approach of Liu et al. (2025) — LcxNet-Fusion (43% reduction
      vs ResNet-18).
    - Batch normalization after each convolution stabilises training
      and allows higher learning rates (Ioffe & Szegedy, 2015).

Architecture summary (128x128x3 input):
    Block 1: Conv2d(3->32, 3x3) → BN → ReLU → MaxPool(2x2)    → 64x64x32
    Block 2: DepthwiseSep(32→64)  → BN → ReLU → MaxPool(2x2)  → 32x32x64
    Block 3: DepthwiseSep(64→128) → BN → ReLU → MaxPool(2x2)  → 16x16x128
    Block 4: DepthwiseSep(128→256) → BN → ReLU → MaxPool(2x2) → 8x8x256
    GlobalAvgPool → 256
    Dropout(0.4) → Linear(256, num_classes)

Approximate parameter count: ~135K (vs 11.2M ResNet-18, 2.2M MobileNetV2)
"""

import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution block.

    Factorises a standard convolution into:
        1. Depthwise conv: one filter per input channel (groups=in_channels)
        2. Pointwise conv: 1x1 convolution to mix channels

    This reduces parameters from (K² x C_in x C_out) to
    (K² x C_in + C_in x C_out), an approximately K²-fold reduction
    for large channel counts.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
        kernel_size:  Spatial kernel size for the depthwise step.
        stride:       Stride for the depthwise step.
        padding:      Padding for the depthwise step.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False,
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class GNSSInterferenceCNN(nn.Module):
    """
    Lightweight 4-block CNN for GNSS interference classification.

    Designed to be trained from scratch (no ImageNet pretraining) on
    128x128 STFT spectrograms. The architecture is deliberately simple
    to enable straightforward ONNX/TensorRT export for Jetson Nano
    deployment without unsupported operations.
    """

    def __init__(self, num_classes: int = 11, in_channels: int = 3,
                 dropout: float = 0.4):
        """
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels (3 for RGB-replicated
                         spectrograms, 1 for single-channel).
            dropout:     Dropout probability before the classifier.
        """
        super().__init__()

        # Block 1: Standard convolution (small input channels, DSC not
        # beneficial at 3 input channels — overhead exceeds savings)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Blocks 2-4: Depthwise separable convolutions
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.block4 = nn.Sequential(
            DepthwiseSeparableConv(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Classifier: GAP eliminates spatial dimensions without
        # flattening, producing a 256-d vector regardless of input size
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # (B, 256, H, W) → (B, 256, 1, 1)
            nn.Flatten(),               # (B, 256, 1, 1) → (B, 256)
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x


def build_custom_cnn(num_classes: int, freeze_backbone: bool = False,
                     dropout: float = 0.4) -> nn.Module:
    """
    Build the custom lightweight CNN.

    Args:
        num_classes:      Number of output classes (e.g. 11).
        freeze_backbone:  Accepted for interface compatibility with the
                          other builders, but ignored — this model is
                          trained from scratch (no pretrained weights).
        dropout:          Dropout probability before the final linear layer.

    Returns:
        GNSSInterferenceCNN nn.Module on CPU.
    """
    if freeze_backbone:
        print("  [Note] freeze_backbone ignored for custom CNN "
              "(no pretrained weights)")

    model = GNSSInterferenceCNN(num_classes=num_classes, dropout=dropout)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Custom CNN: {total:,} total params, {trainable:,} trainable")

    return model


if __name__ == "__main__":
    import torch
    model = build_custom_cnn(11)
    dummy = torch.randn(1, 3, 128, 128)
    out = model(dummy)
    print(f"  Output shape: {out.shape}")
    print(model)