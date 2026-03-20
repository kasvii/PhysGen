import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'physdec'))
sys.path.append(os.path.join(parent_dir, 'dragdec'))
sys.path.append(os.path.join(parent_dir, 'craftsman'))

__all__ = [
    "MultiDecoderModel",
    "MultiTaskSystem",
    "MultiTaskDataModule"
]
