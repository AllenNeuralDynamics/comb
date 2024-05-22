# This class combines a OphysPlaneDataset and BehaviorDatase to create a BehaviorOphysDataset,
# inputs are raw_folder_path and processed_folder_path
# set at

from comb.behavior_session_dataset import BehaviorSessionDataset
from comb.ophys_plane_dataset import OphysPlaneDataset

from comb.utils.dataframe_utils import df_col_to_array

from typing import Union, Optional
from pathlib import Path
import numpy as np


class BehaviorOphysDataset:
    """A class to combine an OphysPlaneDataset and a BehaviorDataset 
    into a single object.

    All attributes of the Other Dataset classes are available as attributes of this class.

    Example #1:
    Assuming the local folders are stuctured like CodeOcean data assets.

    from data_objects.behavior_ophys_dataset import BehaviorOphysDataset
    processed_path = "/allen/programs/mindscope/workgroups/learning/mattd/co_dev/data/1299958728/processed/"
    plane_folder_path = processed_path + "/1299958728"
    raw_path = "/allen/programs/mindscope/workgroups/learning/mattd/co_dev/data/1299958728/raw"
    bod = BehaviorOphysDataset(raw_path, plane_folder_path)

    Example #2:
    Assume raw and processed data assets are attached to capsule.

    """
    def __init__(self,
                plane_folder_path: Union[str, Path],
                raw_folder_path: Union[str, Path],
                verbose: Optional[bool] = False):

        self.ophys_plane_dataset = OphysPlaneDataset(plane_folder_path=plane_folder_path,raw_folder_path=raw_folder_path,verbose=verbose)
        self.behavior_dataset = BehaviorSessionDataset(raw_folder_path=raw_folder_path)

    def __getattr__(self, name):
        if hasattr(self.ophys_plane_dataset, name):
            return getattr(self.ophys_plane_dataset, name)
        elif hasattr(self.behavior_dataset, name):
            return getattr(self.behavior_dataset, name)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # method to print out the attributes the two datasets, remove methods and private attributes
    def print_attr(self):
        attributes = []
        for name, dataset in vars(self).items():
            attributes.append(list(vars(dataset).keys()))
            # for attrb in dir(dataset):
            #     if not attrb.startswith("_"):
            #         attributes.append(attrb)
        return attributes


class BehaviorMultiplaneOphysDataset:
    """A class to combine multiple BehaviorOphysDataset objects into a single object.
        """
    def __init__(self, session_folder_path: Union[str,Path], raw_folder_path: Union[str, Path]):
        self.session_folder_path = Path(session_folder_path)
        self.raw_folder_path = Path(raw_folder_path)
        self.ophys_datasets = {}
        self._get_ophys_datasets()
        self.behavior_dataset = BehaviorSessionDataset(raw_folder_path=raw_folder_path)

    def _get_ophys_datasets(self):
        for plane_folder in self.session_folder_path.glob("*"):
            if plane_folder.is_dir() and not plane_folder.stem.startswith("nextflow"):
                opid = plane_folder.stem
                self.ophys_datasets[opid] = OphysPlaneDataset(plane_folder, raw_folder_path=self.raw_folder_path)

    def __getattr__(self, name):
        if hasattr(self.datasets, name):
            return getattr(self.datasets, name)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


    def all_traces_array(self, traces_key = "dff", return_roi_names = False, remove_nan_rows: Optional[bool] = True):
        """

        Parameters
        ----------
        traces_key : str, optional
            The key to access the traces
            options are ["dff", "events", "filtered_events"] TODO: add raw, demix, etx
            by default "dff"
        return_
        
        """

        traces_list = []
        roi_names_list = []

        if traces_key == "dff":
            attrb_key = "dff_traces"
        elif traces_key == "events" or traces_key == "filtered_events":
            attrb_key = "events"
        # TODO: nueropile, raw, corrected etc.

        for opid, dataset in self.ophys_datasets.items():
            try: 
                traces_df = getattr(dataset, attrb_key)
                traces_array = df_col_to_array(traces_df, traces_key)
                roi_names = traces_df.index # Is this csid or crid?
                if remove_nan_rows:
                    nan_rows = np.isnan(traces_array).all(axis=1)
                    traces_array = traces_array[~nan_rows]
                    roi_names = roi_names[~nan_rows]

                roi_names_list.append(roi_names)
                traces_list.append(traces_array)
            except TypeError:
                print(f"{traces_key} not found for: {opid}")
                continue
        if return_roi_names:
            return np.concatenate(traces_list, axis=1), np.concatenate(roi_names_list)
        else:
            return np.vstack(traces_list)
