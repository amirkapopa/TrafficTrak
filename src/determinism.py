"""Seed every source of randomness used by the pipeline."""

from __future__ import annotations

import os
import random

import cv2
import numpy as np


def set_determinism(seed: int = 0) -> None:
    """Fix RNG seeds.  The pipeline itself has no stochastic components; this
    guards third-party code (OpenCV, NumPy) against accidental randomness."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
