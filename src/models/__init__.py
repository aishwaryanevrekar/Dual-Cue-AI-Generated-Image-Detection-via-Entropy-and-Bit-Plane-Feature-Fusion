"""
LOTA Model Architecture Package.
"""

from src.models.lota import TopKLOTAExtractor
from src.models.classifier import LOTAClassifier

__all__ = ["TopKLOTAExtractor", "LOTAClassifier"]
