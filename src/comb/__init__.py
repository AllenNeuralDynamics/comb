"""Simple package to demo project structure.
"""
__version__ = "0.1.10"

from .behavior_ophys_dataset import BehaviorOphysDataset, BehaviorMultiplaneOphysDataset
from .ophys_plane_dataset import OphysPlaneDataset
from .behavior_session_dataset import BehaviorSessionDataset

__all__ = [
    "BehaviorOphysDataset",
    "BehaviorMultiplaneOphysDataset",
    "BehaviorSessionDataset",
    "OphysPlaneDataset"
]

def one(x):
    if isinstance(x, str):
        return x
    try:
        xlen = len(x)
    except TypeError:
        return x
    if xlen != 1:
        raise OneResultExpectedError("Expected length one result, received: "
                                     f"{x} results from query")
    if isinstance(x, set):
        return list(x)[0]
    else:
        return x[0]

class OneResultExpectedError(RuntimeError):
    pass