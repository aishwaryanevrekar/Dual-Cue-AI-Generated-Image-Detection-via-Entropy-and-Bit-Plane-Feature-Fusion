"""
LOTA Steganalysis Neural Network Classifier.
Exclusively uses LOTA least-significant bit (LSB) noise maps and 4-directional MGPS Top-K quadrant patches.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.lota import TopKLOTAExtractor
from src.utils.logger import get_logger

logger = get_logger("lota_classifier")


class LOTASteganalysisBackbone(nn.Module):
    """
    Deep Convolutional Feature Extractor operating on LOTA stacked noise patch tensors (B, 12, 32, 32).
    """
    def __init__(self, in_channels: int = 12, feature_dim: int = 256):
        super().__init__()
        
        # Layer 1: 12 -> 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),  # -> 16x16
        )

        # Layer 2: 32 -> 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),  # -> 8x8
        )

        # Layer 3: 64 -> 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),  # -> 4x4
        )

        # Layer 4: 128 -> feature_dim
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, feature_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> 1x1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return out.view(out.size(0), -1)


class LOTAClassifier(nn.Module):
    """
    End-to-End LOTA AI-Generated Image Detection Classifier.
    Takes raw image tensors (B, 3, 256, 256) or pre-extracted noise tensors (B, 12, 32, 32),
    extracts LOTA LSB noise features, and predicts Real (0) vs Fake (1) logits.
    """
    def __init__(
        self,
        k_patches: int = 4,
        patch_size: int = 32,
        grid_size: int = 8,
        feature_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.k_patches = k_patches
        self.patch_size = patch_size
        self.grid_size = grid_size

        # LOTA Feature Extractor
        self.lota_extractor = TopKLOTAExtractor(
            k_patches=k_patches,
            patch_size=patch_size,
            grid_size=grid_size,
        )

        # Steganalysis CNN Backbone
        in_channels = k_patches * 3  # 4 * 3 = 12 channels
        self.backbone = LOTASteganalysisBackbone(in_channels=in_channels, feature_dim=feature_dim)

        # Classification Head
        self.classifier_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, 256, 256) float in [0.0, 255.0] or (B, 12, 32, 32).
            return_dict: Whether to return full output dictionary including LOTA intermediate features.

        Returns:
            Logit tensor of shape (B, 1) or dictionary with logits and LOTA features.
        """
        if x.ndim == 4 and x.shape[1] == 3:
            lota_out = self.lota_extractor(x)
            noise_tensor = lota_out["noise_tensor"]
        elif x.ndim == 4 and x.shape[1] == self.k_patches * 3:
            noise_tensor = x
            lota_out = {}
        else:
            raise ValueError(f"Invalid input tensor shape for LOTAClassifier: {x.shape}")

        features = self.backbone(noise_tensor)
        logits = self.classifier_head(features)

        if return_dict:
            res = dict(lota_out)
            res["features"] = features
            res["logits"] = logits
            return res

        return logits
