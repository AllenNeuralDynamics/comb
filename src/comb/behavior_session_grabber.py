import  os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from typing import Any, Optional
import json

from . import file_handling


class BehaviorSessionGrabber(object):
    def __init__(self, 
                 raw_folder_path: str = None,
                 oeid: Optional[str] = None,
                 data_path: Optional[str] = None,
                 load_options: Optional[dict] = None):
        
        self.load_options = load_options

        assert raw_folder_path or oeid is not None, "Must provide either plane_folder_path or oeid"

        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path('../data')

        if oeid:
            self.oeid = oeid
            self.plane_folder_path = self._find_plane_folder_from_oeid(oeid)

        self.raw_folder_path = Path(raw_folder_path)
        self.file_parts = {"platform_json": "_platform.json"}
        self.file_paths = file_handling.get_file_paths_dict(self.file_parts, self.raw_folder_path)

        self.raw_folder_path = Path(raw_folder_path)
        self.sync_file = file_handling.get_sync_file_path(self.raw_folder_path)
        self.stimulus_pkl = self._get_pkl_file()

    def _find_plane_folder_from_oeid(self, oeid):
        # find in results
        found = list(self.data_path.glob(f'**/{oeid}'))
        assert found != 1, f"Found {len(found)} folders with oeid {oeid}"
        return found[0]

    def _get_pkl_file(self):
        with open(self.file_paths['platform_json'], 'r') as f:
            platform_json = json.load(f)

        ophys_path = file_handling.check_ophys_folder(self.raw_folder_path)
        stimulus_pkl_path = ophys_path / platform_json['stimulus_pkl']

        self.file_paths["stimulus_pkl"] = stimulus_pkl_path