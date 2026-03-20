"""
DragDec: Drag Coefficient Estimation Module

This module provides tools for predicting drag coefficients from 3D shape representations.

Key Components:
- DragEstimator: Main model for drag coefficient prediction
- DragDataModule: Data loading and preprocessing (DrivAerNet+ compatible)
- Training and evaluation utilities

Usage:
    from dragdec import DragEstimator, DragDataModule
"""

from .drag_net import DragEstimator, DragCoefficientDecoder
from .drag_dataset import (
    DragDataset, 
    DragDataModule, 
    DragDataModuleConfig
)

__all__ = [
    "DragEstimator",
    "DragCoefficientDecoder", 
    "DragDataset",
    "DragDataModule",
    "DragDataModuleConfig"
]
