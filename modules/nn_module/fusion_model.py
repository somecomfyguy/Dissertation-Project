"""
Spectrogram + statistical feature fusion model for GNSS interference
classification.

Architecture:
    ┌──────────────┐     ┌──────────────┐
    │ Spectrogram  │     │ 8-element    │
    │ 128x128x3    │     │ feature vec  │
    └──────┬───────┘     └──────┬───────┘
           │                    │
    CNN backbone          Feature MLP
    (frozen/unfrozen)     (64 → 32 ReLU)
           │                    │
    Global pool → N-d          32-d
           │                    │
           └────────┬───────────┘
                    │
              Concatenate (N+32)-d
                    │
              Dropout(0.4)
              Linear → num_classes

The CNN backbone is any of the existing model builders. The feature
MLP is a small two-layer network that projects the 8-element statistical
vector into a learned embedding space before concatenation.

This design is motivated by:
    - Contreras Franco et al. (2024): statistical summaries of STFT
      complement spectrogram features on low-resource systems
"""

import torch
import torch.nn as nn
from torchvision import models

from modules.common.types import N_FEATURES


class FeatureMLP(nn.Module):
    """
    Small MLP that projects the 8-element statistical feature vector
    into a learned embedding before fusion with CNN features.
    """

    def __init__(self, input_dim: int = N_FEATURES,
                 hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
    

class FusionModel(nn.Module):
    """
    Dual-input model combining a CNN backbone (spectrogram) with a
    feature MLP (statistical vector).
    """

    def __init__(self, backbone: nn.Module, backbone_out_dim: int,
                 num_classes: int = 11, feat_hidden: int = 64,
                 feat_embed: int = 32, dropout: float = 0.4):
        """
        Args:
            backbone:          CNN with its original classifier head removed.
            backbone_out_dim:  Dimensionality of the backbone's output
                               (e.g. 512 for ResNet-18, 1280 for MobileNetV2).
            num_classes:       Number of output classes.
            feat_hidden:       Hidden layer size in the feature MLP.
            feat_embed:        Output embedding size of the feature MLP.
            dropout:           Dropout before the final classifier.
        """
        super().__init__()
        self.backbone  = backbone
        self.feat_mlp  = FeatureMLP(N_FEATURES, feat_hidden, feat_embed)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(backbone_out_dim + feat_embed, num_classes),
        )

    def forward(self, spectrogram, features):
        cnn_out  = self.backbone(spectrogram)    # (B, backbone_out_dim)
        feat_out = self.feat_mlp(features)       # (B, feat_embed)
        combined = torch.cat([cnn_out, feat_out], dim=1)
        return self.classifier(combined)


def _make_backbone_resnet18(freeze: bool = True):
    """ResNet-18 backbone with classifier head replaced by GAP → flat."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    out_dim = model.fc.in_features     # 512
    model.fc = nn.Identity()           # remove classifier, keep GAP
    return model, out_dim


def _make_backbone_mobilenetv2(freeze: bool = True):
    """MobileNetV2 backbone with classifier removed."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    out_dim = model.classifier[1].in_features   # 1280
    model.classifier = nn.Identity()
    return model, out_dim


def _make_backbone_efficientnetb0(freeze: bool = True):
    """EfficientNet-B0 backbone with classifier removed."""
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    out_dim = model.classifier[1].in_features   # 1280
    model.classifier = nn.Identity()
    return model, out_dim


FUSION_BACKBONES = {
    "resnet18":       _make_backbone_resnet18,
    "mobilenetv2":    _make_backbone_mobilenetv2,
    "efficientnetb0": _make_backbone_efficientnetb0,
}


def build_fusion_model(backbone_name: str = "resnet18",
                       num_classes: int = 11,
                       freeze_backbone: bool = True,
                       dropout: float = 0.4) -> FusionModel:
    """
    Build a fusion model with the specified CNN backbone.

    Args:
        backbone_name:    One of 'resnet18', 'mobilenetv2', 'efficientnetb0'.
        num_classes:      Number of output classes.
        freeze_backbone:  If True, freeze CNN backbone weights.
        dropout:          Dropout before the final classifier.

    Returns:
        FusionModel on CPU.
    """
    if backbone_name not in FUSION_BACKBONES:
        raise ValueError(f"Unknown backbone: {backbone_name}. "
                         f"Choose from: {list(FUSION_BACKBONES.keys())}")

    backbone, out_dim = FUSION_BACKBONES[backbone_name](freeze_backbone)

    model = FusionModel(
        backbone=backbone,
        backbone_out_dim=out_dim,
        num_classes=num_classes,
        dropout=dropout,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Fusion ({backbone_name}): {total:,} total params, "
          f"{trainable:,} trainable")

    return model