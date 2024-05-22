import  os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from typing import Any, Optional
import json


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

        # processed filepaths dict
        self.file_parts = {"platform_json": "_platform.json"} #might not need
        self.file_paths = {}
        self._get_file_path_dict()

        self.raw_folder_path = Path(raw_folder_path)
        # self.oeid = self.plane_folder_path.stem # NOT SURE FOR BEHAVIOR
        self.sync_file = self._get_sync_file()
        self.stimulus_pkl = self._get_pkl_file()

    def _find_plane_folder_from_oeid(self, oeid):
        # find in results
        found = list(self.data_path.glob(f'**/{oeid}'))
        assert found != 1, f"Found {len(found)} folders with oeid {oeid}"
        return found[0]

    def _find_data_file(self, file_part):
        # find in plane_folder_path
        try:
            file = list(self.raw_folder_path.glob(f'**/*{file_part}'))[0]
        except IndexError:
            file = None
        return file
    
    def _get_sync_file(self):
        # load platform json
        with open(self.file_paths['platform_json'], 'r') as f:
            platform_json = json.load(f)

        sync_file_path = self.raw_folder_path / "pophys" / platform_json['sync_file']
        # stimulus pkl: "stimulus_pkl"
        # load sync h5
        #with open(sync_file_path, 'r') as f:
        #    sync_file = json.load(f)
        # add to file paths dict
        self.file_paths['sync_file'] = sync_file_path

    def _get_pkl_file(self):
        with open(self.file_paths['platform_json'], 'r') as f:
            platform_json = json.load(f)

        stimulus_pkl_path = self.raw_folder_path / "pophys" / platform_json['stimulus_pkl']
        # stimulus pkl: "stimulus_pkl"
        # load sync h5
        #with open(sync_file_path, 'r') as f:
        #    sync_file = json.load(f)
        # add to file paths dict
        self.file_paths["stimulus_pkl"] = stimulus_pkl_path

    ####################################################################
    # Data files
    ###################################################################

    def _get_file_path_dict(self):
        for key, value in self.file_parts.items():
            self.file_paths[key] = self._find_data_file(value)
        return self.file_paths