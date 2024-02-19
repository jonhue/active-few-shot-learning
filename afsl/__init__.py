"""
*Active Few-Shot Learning* (`afsl`) is a Python package for intelligent active data selection.

## Why Active Data Selection?

## Getting Started

### Installation

---
"""

from afsl.active_data_loader import ActiveDataLoader
from afsl import acquisition_functions, embeddings, model

__all__ = [
    "ActiveDataLoader",
    "acquisition_functions",
    "embeddings",
    "model",
]
__version__ = "0.1.0"
__author__ = "Jonas Hübotter"
__credits__ = "ETH Zurich, Switzerland"