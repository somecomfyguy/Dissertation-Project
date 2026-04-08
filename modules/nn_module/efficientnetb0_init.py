"""
EfficientNet-B0 model initialisation for GNSS interference classification.

Loads an ImageNet-pretrained EfficientNet-B0 backbone and replaces the
final classifier. EfficientNet uses compound scaling (depth, width,
resolution) to achieve better accuracy per FLOP than ResNet or
MobileNet architectures.

EfficientNet-B0 is the smallest variant and sits between ResNet-18 and
MobileNetV2 in terms of parameter count, making it a useful middle
reference point in the model comparison.
"""

import torch.nn as nn
from torchvision import models


def build_efficientnetb0(num_classes: int, freeze_backbone: bool = True,
                         dropout: float = 0.4) -> nn.Module:
    """
    Load ImageNet-pretrained EfficientNet-B0 and replace the classifier head.

    Args:
        num_classes:      Number of output classes (e.g. 11).
        freeze_backbone:  If True, freeze all layers except the classifier.
        dropout:          Dropout probability before the final linear layer.

    Returns:
        EfficientNet-B0 nn.Module on CPU. Call model.to(device) after.
    """
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # EfficientNet classifier is model.classifier (a Sequential):
    #   [0] Dropout(0.2, inplace=True)
    #   [1] Linear(1280, 1000)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  EfficientNet-B0: {total:,} total params, {trainable:,} trainable")

    return model