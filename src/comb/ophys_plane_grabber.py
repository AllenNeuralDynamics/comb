import  os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from typing import Any, Optional,Union
import json

from . import file_handling


class OphysPlaneGrabber(object):
    def __init__(self,
                 plane_folder_path: Union[str, Path] = None,
                 raw_folder_path: Union[str, Path] = None,
                 opid: Optional[str] = None,
                 data_path: Optional[str] = None,
                 verbose=False):

        assert plane_folder_path or opid is not None, "Must provide either plane_folder_path or opid"
        assert raw_folder_path is not None, "Must provide raw_folder_path"

        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path('../data')

        if opid:
            self.opid = opid
            self.plane_folder_path = self._find_plane_folder_from_opid(opid)
        elif plane_folder_path:
            self.plane_folder_path = Path(plane_folder_path)
            self.opid = self.plane_folder_path.stem
        self.verbose = verbose
    
        # processed filepaths dict
        self.processed_file_parts = {"processing_json": "processing.json",
                                     #"params_json": "_params.json",
                                     #"registered_metrics_json": "_registered_metrics.json",
                                     #"output_json": "_output.json",
                                     "average_projection_png": "_average_projection.png",
                                     "max_projection_png": "_maximum_projection.png",
                                     "motion_transform_csv": "_motion_transform.csv",
                                     "decrosstalk_avg_png": "_decrosstalk_avg_img.png",
                                     "decrosstalk_max_png": "_decrosstalk_max_img.png",
                                     "extraction_h5": "extraction.h5",
                                     "dff_h5": "dff.h5",
                                     "events_oasis_h5": "events_oasis.h5"}

        self.raw_file_parts = {"session_json": "session.json",
                               "platform_json": "_platform.json"}

        self.raw_folder_path = Path(raw_folder_path)

        self.sync_file = file_handling.get_sync_file_path(self.raw_folder_path)
        self.file_paths = file_handling.get_file_paths_dict(self.processed_file_parts, self.plane_folder_path)
        self.file_paths.update(file_handling.get_file_paths_dict(self.raw_file_parts, self.raw_folder_path))
        self.file_paths.update({"sync_file": self.sync_file})

    def _find_plane_folder_from_opid(self, opid):
        # find in results
        found = list(self.data_path.glob(f'**/{opid}'))
        assert found != 1, f"Found {len(found)} folders with opid {opid}"
        return found[0]
