# This class combines a OphysPlaneDataset and BehaviorDatase to create a BehaviorOphysDataset,
# inputs are raw_folder_path and processed_folder_path
# set at

from comb.behavior_session_dataset import BehaviorSessionDataset
from comb.ophys_plane_dataset import OphysPlaneDataset
from comb.utils.dataframe_utils import df_col_to_array

from aind_ophys_data_access import metadata

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
                eye_tracking_path: Optional[Union[str, Path]] = None,
                verbose: Optional[bool] = False,
                project_code: Optional[str] = None,
                roi_matching_path: Optional[Union[str, Path]] = None,
                pipeline_version: Optional[str] = None,):

        if not Path(plane_folder_path).exists():
            raise FileNotFoundError(f"Path does not exist: {plane_folder_path}")
        if not Path(raw_folder_path).exists():
            raise FileNotFoundError(f"Path does not exist: {raw_folder_path}")

        self.ophys_plane_dataset = OphysPlaneDataset(plane_folder_path=plane_folder_path,
                                                     raw_folder_path=raw_folder_path,
                                                     roi_matching_path=roi_matching_path, verbose=verbose,
                                                     pipeline_version=pipeline_version)
        self.behavior_dataset = BehaviorSessionDataset(raw_folder_path=raw_folder_path, 
                                                       project_code=project_code, 
                                                       eye_tracking_path=eye_tracking_path)
        
        self.metadata = self._session_metadata()
    
    def _session_metadata(self):
        
        jsons_dict = metadata.load_metadata_json_files(self.raw_folder_path)
        metadata_dict = metadata.metadata_for_multiplane_session(jsons_dict)

        # pop var not needed
        remove_keys = ["ophys_fovs", "microscope_description", "ophys_seg_approach","ophys_seg_descr"]
        for key in remove_keys:
            if metadata_dict.get(key) is not None:
                metadata_dict.pop(key)
    
        if self.ophys_plane_dataset.metadata is not None:
            metadata_dict["plane"] = self.ophys_plane_dataset.metadata
            
        if self.behavior_dataset.metadata is not None:
            metadata_dict["behavior"] = self.behavior_dataset.metadata
            
        # if self.eye_tracking_path is not None:
        #     metadata_dict["eye_tracking"] = self.eye_tracking_dataset.metadata

        metadata_dict["raw_path"] = self.behavior_dataset.raw_folder_path
        metadata_dict["processed_path"] = self.ophys_plane_dataset.plane_folder_path.parent
        metadata_dict["session_name"] = self.behavior_dataset.raw_folder_path.stem
        
        # add ophys_frame_rate for downstream
        metadata_dict["ophys_frame_rate"] = metadata_dict["plane"]["ophys_frame_rate"]
        
        # add imaging_depth from json_dict["session"] # DELETE 
        #metadata_dict["plane"]["imaging_depth"] = {"imaging_depth": jsons_dict["session"]["targeted_depth"]}
        
        # alphabetize keys
        metadata_dict = dict(sorted(metadata_dict.items()))

        return metadata_dict
                

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
    def __init__(self, 
                 session_folder_path: Union[str,Path], 
                 raw_folder_path: Union[str, Path],
                 project_code: Optional[str] = None,
                 eye_tracking_path: Optional[Union[str, Path]] = None,
                 roi_matching_path: Optional[Union[str, Path]] = None,
                 pipeline_version: Optional[str] = None):
        
        self.session_folder_path = Path(session_folder_path)
        self.raw_folder_path = Path(raw_folder_path)
        self.roi_matching_path = roi_matching_path
        self.project_code = project_code
        self.pipeline_version = pipeline_version
        
        if not Path(session_folder_path).exists():
            raise FileNotFoundError(f"Path does not exist: {session_folder_path}")
        if not Path(raw_folder_path).exists():
            raise FileNotFoundError(f"Path does not exist: {raw_folder_path}")

        self.ophys_datasets = {}
        self._get_ophys_datasets()
        self.behavior_dataset = BehaviorSessionDataset(raw_folder_path=raw_folder_path,
                                                       project_code=self.project_code,
                                                       eye_tracking_path=eye_tracking_path)

    def _get_ophys_datasets(self):
        for plane_folder in self.session_folder_path.glob("*"):
            if plane_folder.is_dir() and not plane_folder.stem.startswith("nextflow")\
                                     and 'nwb' not in plane_folder.stem:
                opid = plane_folder.stem
                self.ophys_datasets[opid] = OphysPlaneDataset(plane_folder, 
                                                              raw_folder_path=self.raw_folder_path, 
                                                              roi_matching_path=self.roi_matching_path,
                                                              pipeline_version=self.pipeline_version)

    def __getattr__(self, name):
        if hasattr(self.datasets, name):
            return getattr(self.datasets, name)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


    def all_traces_array(self, 
                         traces_key: str = "dff", 
                         return_roi_names: bool = False, 
                         remove_nan_rows: Optional[bool] = True):
        """

        Parameters
        ----------
        traces_key : str, optional
            The key to access the traces
            options are ["dff", "events", "filtered_events"] 
            by default "dff"
            TODO (maybe): add raw, demix, etx
        return_roi_names : bool, optional
            Whether to return the roi_names as well, by default False
        remove_nan_rows : bool, optional
            Whether to remove rows with all NaNs, by default True
            
        Returns
        -------
        traces_array : np.ndarray
        
        """

        traces_list = []
        roi_names_list = []

        if traces_key == "dff":
            attrb_key = "dff_traces"
        elif traces_key == "events" or traces_key == "filtered_events":
            attrb_key = "events"
        # TODO: nueropil, raw, corrected etc.

        for opid, dataset in self.ophys_datasets.items():
            try: 
                traces_df = getattr(dataset, attrb_key)
                traces_array = df_col_to_array(traces_df, traces_key)
                roi_names = traces_df.index # Is this csid or crid?
                if remove_nan_rows:
                    nan_rows = np.isnan(traces_array).all(axis=1)
                    traces_array = traces_array[~nan_rows]
                    roi_names = roi_names[~nan_rows]
                    roi_names = np.array([f"{opid}_{int(roi):04}" for roi in roi_names])

                roi_names_list.append(roi_names)
                traces_list.append(traces_array)
            except TypeError:
                print(f"{traces_key} not found for: {opid}")
                continue
        if return_roi_names:
            return np.vstack(traces_list), np.concatenate(roi_names_list)
        else:
            return np.vstack(traces_list)
