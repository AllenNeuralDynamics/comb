from typing import Dict, Union, Any
from pathlib import Path

import pandas as pd

from comb.processing.eye_tracking import load_eye_tracking_hdf
    
class EyeTrackingFile(object):
    """A DataFile which contains methods for accessing and loading
    eye tracking data.
    """

    def __init__(self, data: Any, filepath: str):
        self._data = data
        self._filepath = None

    @property
    def data(self) -> Any:  # pragma: no cover
        return self._data

    @property
    def filepath(self) -> str:  # pragma: no cover
        return self._filepath

    @staticmethod
    def load_data(filepath: Union[str, Path]) -> pd.DataFrame:
        return load_eye_tracking_hdf(filepath)
