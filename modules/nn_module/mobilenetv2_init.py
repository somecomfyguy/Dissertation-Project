"""
MobileNetV2 model initialisation for GNSS interference classification.

Loads an ImageNet-pretrained MobileNetV2 backbone and replaces the final
classifier to match the project's class taxonomy. Supports the same
two-phase frozen/unfrozen training strategy as ResNet-18.

MobileNetV2 is the primary embedded deployment candidate due to its
inverted residual architecture and depthwise separable convolutions,
which yield significantly fewer parameters and FLOPs than ResNet-18
while maintaining competitive accuracy.
"""

import torch.nn as nn
from torchvision import models


def build_mobilenetv2(num_classes: int, freeze_backbone: bool = True,
                      dropout: float = 0.4) -> nn.Module:
    """
    Load ImageNet-pretrained MobileNetV2 and replace the classifier head.

    Args:
        num_classes:      Number of output classes (e.g. 11).
        freeze_backbone:  If True, freeze all layers except the classifier.
        dropout:          Dropout probability before the final linear layer.

    Returns:
        MobileNetV2 nn.Module on CPU. Call model.to(device) after.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # MobileNetV2 classifier is model.classifier (a Sequential):
    #   [0] Dropout(0.2)
    #   [1] Linear(1280, 1000)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  MobileNetV2: {total:,} total params, {trainable:,} trainable")

    return model