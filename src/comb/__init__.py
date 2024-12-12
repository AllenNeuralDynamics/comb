"""Simple package to demo project structure.
"""
__version__ = "0.1.6"

from .behavior_ophys_dataset import BehaviorOphysDataset, BehaviorMultiplaneOphysDataset
from .ophys_plane_dataset import OphysPlaneDataset
from .behavior_session_dataset import BehaviorSessionDataset

__all__ = [
    "BehaviorOphysDataset",
    "BehaviorMultiplaneOphysDataset",
    "BehaviorSessionDataset",
    "OphysPlaneDataset"
]