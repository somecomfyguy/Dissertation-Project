"""
Function to build resnet18
"""
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int, freeze_backbone: bool = True):
    """
    Load ImageNet-pretrained ResNet-18 and replace the final FC layer.
 
    Args:
        num_classes:      Number of output classes
        freeze_backbone:  If True, freeze all layers except the final FC.
                          Set to False for full fine-tuning.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
 
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
 
    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
 
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ResNet-18: {total:,} total params, {trainable:,} trainable")
 
    return model