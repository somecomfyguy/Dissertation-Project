"""
ResNet-18 model initialisation for GNSS interference classification.

Loads an ImageNet-pretrained ResNet-18 backbone and replaces the final
fully-connected layer to match the project's class taxonomy (12 classes
by default). Supports a two-phase training strategy: backbone frozen
initially, then fully unfrozen for fine-tuning.
"""
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int, freeze_backbone: bool = True):
    """
    Load ImageNet-pretrained ResNet-18 and replace the final FC layer.
 
    Args:
        num_classes:      Number of output classes (e.g. 12 for the unified
                          OAKBAT + Swinney taxonomy).
        freeze_backbone:  If True, freeze all layers except the final FC layer.
                          Set to False for full end-to-end fine-tuning.

    Returns:
        model:  ResNet-18 nn.Module with a replaced FC head. Parameters are
                on CPU; call model.to(device) after this function.
    """
    # Init model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
 
    # Autograd stops recording operations on frozen parameters,
    # so only the FC head receives gradient updates in the frozen phase.
    if freeze_backbone:                     
        for param in model.parameters():
            param.requires_grad = False  
 
    # Replace the ImageNet 1000-class head with a task-specific head.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
 
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ResNet-18: {total:,} total params, {trainable:,} trainable")
 
    return model

if __name__ == "__main__":
    # Sanity check to make sure model is initiallised correctly
    model = build_resnet18(12)
    print(model)