"""
Unit tests for LOTAClassifier model architecture, forward pass, and training step.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.models.classifier import LOTAClassifier, LOTASteganalysisBackbone


def test_lota_steganalysis_backbone():
    """Verify LOTASteganalysisBackbone processes 12x32x32 noise patch tensor to 256d vector."""
    backbone = LOTASteganalysisBackbone(in_channels=12, feature_dim=256)
    noise_tensor = torch.randn(4, 12, 32, 32, dtype=torch.float32)
    
    features = backbone(noise_tensor)
    assert isinstance(features, torch.Tensor)
    assert features.shape == (4, 256)


def test_lota_classifier_forward():
    """Verify LOTAClassifier forward pass from raw RGB image tensor (B, 3, 256, 256)."""
    model = LOTAClassifier(k_patches=4, patch_size=32, grid_size=8)
    images = torch.randint(0, 256, (2, 3, 256, 256), dtype=torch.float32)
    
    logits = model(images)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 1)


def test_lota_classifier_return_dict():
    """Verify return_dict=True includes LOTA intermediate features."""
    model = LOTAClassifier(k_patches=4, patch_size=32, grid_size=8)
    images = torch.randint(0, 256, (2, 3, 256, 256), dtype=torch.float32)
    
    out_dict = model(images, return_dict=True)
    assert isinstance(out_dict, dict)
    assert "logits" in out_dict
    assert "features" in out_dict
    assert "noise_tensor" in out_dict
    assert out_dict["logits"].shape == (2, 1)
    assert out_dict["noise_tensor"].shape == (2, 12, 32, 32)


def test_lota_classifier_training_step():
    """Verify single optimization step updates parameters with non-zero gradients."""
    model = LOTAClassifier(k_patches=4, patch_size=32, grid_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    images = torch.randint(0, 256, (2, 3, 256, 256), dtype=torch.float32)
    labels = torch.tensor([0.0, 1.0], dtype=torch.float32).unsqueeze(1)

    logits = model(images)
    loss = criterion(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    
    # Check that backbone weights received non-zero gradients
    has_grad = any(p.grad is not None and torch.abs(p.grad).sum() > 0 for p in model.backbone.parameters())
    assert has_grad is True

    optimizer.step()
