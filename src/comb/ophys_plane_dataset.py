from comb.ophys_plane_grabber import OphysPlaneGrabber
from comb.processing.sync.sync_utilities import get_synchronized_frame_times

from comb import metadata

from . import file_handling # TODO change to data_access?

from lamf_analysis import utils as lamf_utils # for motion border

from typing import Any, Optional,Union
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import h5py
import numpy as np
import xarray as xr
from pathlib import Path

from . import data_file_keys

class LazyLoadable(object):
    def __init__(self, name, calculate):
        ''' Wrapper for attributes intended to be computed or loaded once, 
        then held in memory by a containing object.

        Parameters
        ----------
        name : str
            The name of the hidden attribute in which this attribute's data will be stored.
        calculate : fn
            a function (presumably expensive) used to calculate or load this attribute's data

        '''

        self.name = name
        self.calculate = calculate

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if not hasattr(obj, self.name):
            setattr(obj, self.name, self.calculate(obj))
        return getattr(obj, self.name)


class OphysPlaneDataset(OphysPlaneGrabber):
    def __init__(self,
                plane_folder_path: Union[str, Path] = None,
                raw_folder_path: Union[str, Path] = None,
                roi_matching_path: Optional[Union[str, Path]] = None,
                opid: Optional[str] = None,
                data_path: Optional[str] = None,
                verbose: Optional[bool] = False,
                pipeline_version: Optional[str] = None,
                apply_patch: Optional[bool] = True):
        super().__init__(plane_folder_path=plane_folder_path,
                         raw_folder_path=raw_folder_path,
                         opid=opid,
                         data_path=data_path,
                         verbose=verbose)
        self.pipeline_version = pipeline_version
        self.metadata = self._set_metadata()
        
        self._add_plane_order_index()
        self._set_metadata_from_jsons()
        

        # keep for legacy purposes
        self.ophys_experiment_id = self._resolve_ophys_experiment_id()

        # no opid is given, set as= ophys_experiment_id
        if self.opid is None:
            self.opid = self.ophys_experiment_id


        # currently pipeline give all nan traces; lets remove
        try:
            _ = self.get_cell_specimen_table()
            self._set_all_nan_traces_invalid()
        except TypeError: # No dff_file
            pass

        self.metadata['ophys_frame_rate'] = self._get_ophys_frame_rate()
        
        if roi_matching_path is not None:
            self.file_paths['roi_matching_table'] = roi_matching_path
            self.roi_matching_table = self._load_roi_matching_table()

        # Patch some of the issues, until they are resolved.
        # 1. ROI filtering - will be done by ROICat
        if apply_patch:
            self._patch_attributes()

    ####################################################################
    # Data files
    ####################################################################
    
    
    def _infer_plane_sort_index(self):
        """The names of the plane folders, sorted, determine the order of the planes
        
        In some cases this is used for container assignment.
        
        Returns
        -------
        plane_folder_index_map : dict
        """
        
        if self.pipeline_version == 'v6':
            
            # TODO: getting plane folders and validating can be elsehwere
            # TODO: look up from brain areas?
            valid_prefix = ["VISp", "VISl", "VISal", "VISam", "VISpm", "VISrl"]
            
            processed_path = self.metadata["plane_path"].parent
            plane_folders = [f for f in processed_path.iterdir() if f.is_dir() and '.nwb' not in f.name \
                and 'vasculature' not in f.name \
                and 'matched' not in f.name]
            
            assert all([f.name.split('_')[1].isdigit() for f in plane_folders]), "Plane folders are not named as expected"
            assert all([f.name.split('_')[0] in valid_prefix for f in plane_folders]), "Plane folders are not named as expected"
            plane_folders = sorted(plane_folders, key=lambda x: x.name)
            plane_folder_index_map = {f.name: int(f.name.split('_')[1]) for f in plane_folders}

        elif self.pipeline_version == 'v4-from_lims':
            # generally else = v4, specifically for saffron sessions
            
            processed_path = self.metadata["plane_path"].parent
            plane_folders = [f for f in processed_path.iterdir() if f.is_dir()]
            
            # assert 8 folders
            assert len(plane_folders) == 8, f"Expected 8 plane folders, found {len(plane_folders)}"
            plane_folders = sorted(plane_folders, key=lambda x: x.name)
            plane_folder_index_map = {plane_folder.name: i for i, plane_folder in enumerate(plane_folders)}
        
        else:
            raise NotImplementedError(f"Pipeline version {self.pipeline_version} not supported")

        return plane_folder_index_map
    
    
    def _add_plane_order_index(self):
        plane_order_map = self. _infer_plane_sort_index()
        self.metadata['plane_order_index'] = plane_order_map[self.plane_folder_path.name]

    
    def _load_roi_matching_table(self):
        """ Load ROI matching table from roi_matching_path
        
        To get the right container/plane index, we need to infer the order of the planes.
        Saffron was generating the planes in the order of the folders in the processed_path.
        Later versions of the pipeline may not follow this order, will have to adapt the code.
        """
        plane_name = self.plane_folder_path.name
        # assert "721291" in  str(self.plane_folder_path), "ROI matches only for 721291 (11/6/2024)"
        plane_container_map = self._infer_plane_order()
        container_index = plane_container_map[plane_name]
        matching_path = Path(self.file_paths['roi_matching_table']) / str(container_index) / 'ROICaT.tracking.results.csv'
        
        match_table = pd.read_csv(matching_path)
        match_table = match_table[match_table.fov_name.astype(str) == str(plane_name)].reset_index(drop=True)
        
        return match_table
        

    def _resolve_ophys_experiment_id(self):
        if self.plane_folder_path is not None:
            ophys_experiment_id = self.plane_folder_path.name
        elif self.opid is not None:
            ophys_experiment_id = self.opid

        return ophys_experiment_id


    def _parse_mesoscope_metadata(self):
        assert self.file_paths['session_json'] is not None, "session.json is not found, only mesoscope data is supported"

        split_dict = {}
        with open(self.file_paths['session_json']) as json_file:
            split_json = json.load(json_file)
        data_stream = split_json['data_streams'][0]
        # Assign plane_group_count
        num_groups_inferred = len(data_stream['ophys_fovs']) // 2
        max_coupled_fov_index = np.max([int(fov['coupled_fov_index']) for fov in data_stream['ophys_fovs']])
        assert num_groups_inferred == max_coupled_fov_index + 1, "Number of groups inferred from session.json does not match number of groups in splitting json"
        split_dict['plane_group_count'] = num_groups_inferred

        # Find the plane information from session.json by comparing with opid
        all_plane_names = np.array([f'{fov["targeted_structure"]}_{fov["index"]}' for fov in data_stream['ophys_fovs']])
        # Find a unique match
        matched_inds = np.where(all_plane_names == self.opid)[0]
        assert len(matched_inds) == 1, "Could not find a unique match for plane in session.json"
        matched_index = matched_inds[0]
        
        # Assign plane_group_index (which is the same as copuled_fov_index)
        split_dict['plane_group_index'] = data_stream['ophys_fovs'][matched_index]['coupled_fov_index']
        split_dict['split_json_scanfield_z'] = data_stream['ophys_fovs'][matched_index]['scanfield_z']

        return split_dict

    def _get_ophys_frame_rate(self):

        dt = self.ophys_timestamps.diff().mean()
        frame_rate = 1/dt

        if self.verbose:
            print("Calculating frame rate from ophys_timestamps, not metadata")

        return frame_rate

    def _set_metadata(self):
        metadata_content = {}
        with open(self.file_paths['platform_json']) as json_file:
            platform = json.load(json_file)

        split_dict = self._parse_mesoscope_metadata()
        metadata_content.update(split_dict)

        metadata_content['plane_path'] = self.plane_folder_path
        
        plane_folder_name = self.plane_folder_path.name
        session_name = self.plane_folder_path.parent.name
        subject_id = session_name.split("_")[1]
        date = session_name.split("_")[2]
        
        # for this plane set _inferred_plane_order
        metadata_content['plane_session_key'] = f"{subject_id}_{date}_{plane_folder_name}"
        
        # TODO: should get all metadata from jsons or docdb.
        # # open session.json
        # with open(self.file_paths['session_json']) as json_file:
        #     session = json.load(json_file)
        
        return metadata_content

    
    def _set_metadata_from_jsons(self):
        json_dicts = metadata.load_metadata_json_files(self.raw_folder_path)
        
        index_key_only = True if self.pipeline_version == 'v4-from_lims' else False
        ophys_fovs_dict = metadata.extract_ophys_fovs(json_dicts["session"], 
                                                      index_key_only = index_key_only)

        # using the inferred plane order, grab the right plane_metadata
        if self.pipeline_version == 'v4-from_lims':
            plane_order_map = self._infer_plane_sort_index()
            plane_order_index = plane_order_map[self.plane_folder_path.name]
            fov_metadata = ophys_fovs_dict[plane_order_index]
        elif self.pipeline_version == 'v6':
            fov_metadata = ophys_fovs_dict[self.plane_folder_path.name]
        self.metadata.update(fov_metadata)
        

    def _set_all_nan_traces_invalid(self):
        ''' Only possible when it has _cell_specimen_table as an attribute.
        Ignore otherwise'''
        if hasattr(self, '_cell_specimen_table') and (self._cell_specimen_table is not None):
            dff = self.dff_traces
            nan_ids = []
            # iterate dff.dff, check if array is all nan
            for cell_specimen_id, trace in dff.dff.items(): # TODO: remove iteration
                if np.all(np.isnan(trace)):
                    nan_ids.append(cell_specimen_id)
            if len(nan_ids) > 0:
                new_csid_table = self._cell_specimen_table
                new_csid_table.loc[nan_ids, 'valid_roi'] = False

                # for each nan_ids, set append 'nan trace' to exclusion_labels cell
                for cell_specimen_id in nan_ids:
                    if new_csid_table.loc[cell_specimen_id, 'exclusion_labels'] is None:
                        new_csid_table.loc[cell_specimen_id, 'exclusion_labels'] = ['nan trace']

                self._cell_specimen_table = new_csid_table

                if self.verbose:
                    print(f"Set {len(nan_ids)} cell_specimen_ids to invalid_roi, found all nan traces")
        

    def _add_csid_to_table(self, table):
        """Cell specimen ids are not avaiable in CodeOcean, as they were in LIMS (01/18/2024)
        Use this method to add them.

        Option 1: duplicated cell_roi_id
        Currently, cell_roi_ids are just indexes. Eventually they will be given numbers as well.
        """

        # Option 1: just duplicated cell_roi_id
        # check table index name
        if table.index.name == 'cell_roi_id':
            table['cell_specimen_id'] = table.index.values
        elif 'cell_roi_id' in table.columns:
            table['cell_specimen_id'] = table.cell_roi_id
        else:
            raise Exception('Table does not contain cell_roi_id')
        table = table.set_index('cell_specimen_id')

        return table

    def get_average_projection_png(self):
        self._average_projection = plt.imread(self.file_paths['average_projection_png'])
        return self._average_projection

    def get_max_projection_png(self):
        self._max_projection = plt.imread(self.file_paths['max_projection_png'])
        return self._max_projection

    def get_motion_transform_csv(self):
        self._motion_transform = pd.read_csv(self.file_paths['motion_transform_csv'])
        return self._motion_transform
    
    def roi_table_from_mask_arrays(pixel_masks: np.ndarray):
        
        # assert 3d
        columns = ['mask_matrix',
                    'height',
                    'width',
                    'X',
                    'Y',
                    'centroid',
                    'bounding_box',
                    'valid_roi',
                    'exclusion_labels']

        roi_table = pd.DataFrame(index=range(pixel_masks.shape[0]), columns=columns)
        for i in range(pixel_masks.shape[0]):
            roi_mask = pixel_masks[i]
            roi_table.loc[i, 'mask_matrix'] = roi_mask
        
            # find a bounding box around roi
            non_zero_coords = np.array(np.where(roi_mask > 0))  # Shape (2, N) where N = number of non-zero points

            # Get the bounds of the bounding box
            min_row, min_col = non_zero_coords.min(axis=1)
            max_row, max_col = non_zero_coords.max(axis=1)

            # Bounding box coordinates
            bounding_box = (min_row, min_col, max_row, max_col)
            height = max_row - min_row
            width = max_col - min_col

            roi_table.loc[i, 'bounding_box'] = [bounding_box]
            roi_table.loc[i, 'height'] = height
            roi_table.loc[i, 'width'] = width
            roi_table.loc[i, 'X'] = min_col
            roi_table.loc[i, 'Y'] = min_row
            roi_table.loc[i, 'centroid'] = (min_col + width / 2, min_row + height / 2)
            
            # legacy attributes
            roi_table.loc[i, 'valid_roi'] = True
            roi_table.loc[i, 'exclusion_labels'] = None
                
        return roi_table

    # TODO: should we rename the attribute to segmentation? (MJD)
    def get_cell_specimen_table(self):
        if hasattr(self, '_cell_specimen_table') and (self._cell_specimen_table is not None):
            return self._cell_specimen_table
        else:
            pixel_masks = file_handling.load_sparse_array(self.file_paths['extraction_h5'])

            def _roi_table_from_mask_arrays(pixel_masks: np.ndarray,
                                        index_name: str = 'id'):
                ''' Copied from capsule_data_utils.roi_table_from_mask_arrays
                due to circular import issue '''
                columns = ['mask_matrix',
                            'height',
                            'width',
                            'x',
                            'y',
                            'centroid',
                            'bounding_box',
                            'valid_roi',
                            'exclusion_labels']

                roi_table = pd.DataFrame(index=range(pixel_masks.shape[0]), columns=columns)
                for i in range(pixel_masks.shape[0]):
                    roi_mask = pixel_masks[i]
                    roi_table.loc[i, 'mask_matrix'] = roi_mask
                
                    # find a bounding box around roi
                    non_zero_coords = np.array(np.where(roi_mask > 0))  # Shape (2, N) where N = number of non-zero points

                    # Get the bounds of the bounding box
                    min_row, min_col = non_zero_coords.min(axis=1)
                    max_row, max_col = non_zero_coords.max(axis=1)

                    # Bounding box coordinates
                    bounding_box = (min_row, min_col, max_row, max_col)
                    height = max_row - min_row
                    width = max_col - min_col

                    roi_table.loc[i, 'bounding_box'] = [bounding_box]
                    roi_table.loc[i, 'height'] = height
                    roi_table.loc[i, 'width'] = width
                    roi_table.loc[i, 'x'] = min_col
                    roi_table.loc[i, 'y'] = min_row
                    roi_table.loc[i, 'centroid'] = (min_col + width / 2, min_row + height / 2)
                    
                    # legacy attributes
                    roi_table.loc[i, 'valid_roi'] = True
                    roi_table.loc[i, 'exclusion_labels'] = None

                # reset and rename col
                roi_table.index.name = index_name
                roi_table = roi_table.reset_index(drop=False)

                return roi_table
            
            roi_table = _roi_table_from_mask_arrays(pixel_masks)
            roi_table = roi_table.rename(columns={'id': 'cell_roi_id'})
            # cell_specimen_table = self._add_csid_to_table(cell_specimen_table)
            # self._set_all_nan_traces_invalid()
            
            self._cell_specimen_table = roi_table
            return self._cell_specimen_table

    def get_raw_fluorescence_traces(self):
        raw_traces, cell_roi_ids = file_handling.load_signals(self.file_paths['extraction_h5'], 
                                                              h5_group="traces", h5_key="roi")
        
        traces_df = pd.DataFrame(index=cell_roi_ids, columns=['raw_fluorescence_traces'])
        for i, crid in enumerate(cell_roi_ids):
            traces_df.loc[crid, 'raw_fluorescence_traces'] = raw_traces[i, :]
#       traces_df = self._add_csid_to_table(traces_df)
        self._raw_fluorescence_traces = traces_df
        return self._raw_fluorescence_traces

    def get_neuropil_traces(self):
        # TODO: cell_roi_ids are removed from this table. Should we add them back?
        # TODO: should we rename this attribute to neuropil_corrected_traces?

        f = h5py.File(self.file_paths['neuropil_correction_h5'], mode='r')
        neuropil_traces_array = np.asarray(f['FC'])
        roi_ids = [int(roi_id) for roi_id in np.asarray(f['roi_names'])]
        RMSE = [value for value in np.asarray(f['RMSE'])]
        r = [value for value in np.asarray(f['r'])]

        # convert to dataframe 
        neuropil_traces = pd.DataFrame(index=roi_ids, columns=['neuropil_fluorescence_traces', 'r', 'RMSE'])
        for i, roi_id in enumerate(roi_ids):
            neuropil_traces.loc[roi_id, 'neuropil_fluorescence_traces'] = neuropil_traces_array[i, :]
            neuropil_traces.loc[roi_id, 'r'] = r[i]
            neuropil_traces.loc[roi_id, 'RMSE'] = RMSE[i]
        neuropil_traces.index.name = 'cell_roi_id'
        neuropil_traces = self._add_csid_to_table(neuropil_traces)
        self._neuropil_traces = neuropil_traces
        return self._neuropil_traces

    def get_neuropil_masks(self):

        with open(self.file_paths['neuropil_masks_json']) as json_file:
            neuropil_mask_data = json.load(json_file)

        neuropil_masks = pd.DataFrame(neuropil_mask_data['neuropils'])
        neuropil_masks = neuropil_masks.rename(columns={'id':'cell_roi_id'})
        neuropil_masks = self._add_csid_to_table(neuropil_masks)
        self._neuropil_masks = neuropil_masks
        return self._neuropil_masks

    def get_neuropil_traces_xr(self):
        """

        Why xarray?
        Labeled indexing, select data by cell_rois_id directly
        Can be more efficient and intuitive than pandas, which uses boolean indexing

        multidimensional labeled: xarray
        tabular with groupby: pandas

        Example:
        if x is xarray, 

        cell_roi_id = 1
        x.sel(cell_roi_id=cell_roi_id).neuropil_fluorescence_traces.plot.line()
        x.sel(cell_roi_id=cell_roi_id).RMSE

        """

        f = h5py.File(self.file_paths['neuropil_correction_h5'], mode='r')
        neuropil_traces_array = np.asarray(f['FC'])
        roi_ids = [int(roi_id) for roi_id in np.asarray(f['roi_names'])]
        RMSE = [value for value in np.asarray(f['RMSE'])]
        r = [value for value in np.asarray(f['r'])]
        f.close()

        neuropil_traces = xr.DataArray(neuropil_traces_array, dims=('cell_roi_id', 'time'), coords={'cell_roi_id': roi_ids, 'time': np.arange(neuropil_traces_array.shape[1])})
        r = xr.DataArray(r, dims=('cell_roi_id',), coords={'cell_roi_id': roi_ids})
        RMSE = xr.DataArray(RMSE, dims=('cell_roi_id',), coords={'cell_roi_id': roi_ids})
        self._neuropil_traces_xr = xr.Dataset({'neuropil_fluorescence_traces': neuropil_traces, 'r': r, 'RMSE': RMSE})

        return self._neuropil_traces_xr

    def get_demixed_traces(self):

        f = h5py.File(self.file_paths['demixing_output_h5'], mode='r')
        demixing_output_array = np.asarray(f['data'])
        roi_ids = [int(roi_id) for roi_id in np.asarray(f['roi_names'])]

        # convert to dataframe 
        demixed_traces = pd.DataFrame(index=roi_ids, columns=['demixed_fluorescence_traces'])
        for i, roi_id in enumerate(roi_ids):
            demixed_traces.loc[roi_id, 'demixed_fluorescence_traces'] = demixing_output_array[i, :]
        demixed_traces.index.name = 'cell_roi_id'
        demixed_traces = self._add_csid_to_table(demixed_traces)
        self._demixed_traces = demixed_traces
        return self._demixed_traces

    # dff_traces where stored differently in LIMS processed data
    # def get_dff_traces(self):

    #     f = h5py.File(self.file_paths['dff_h5'], mode='r')
    #     dff_traces_array = np.asarray(f['data'])
    #     roi_ids = [int(roi_id) for roi_id in np.asarray(f['roi_names'])]
    #     num_small_baseline_frames = [value for value in np.asarray(f['num_small_baseline_frames'])]
    #     sigma_dff = [value for value in np.asarray(f['sigma_dff'])]

    #     # convert to dataframe 
    #     dff_traces = pd.DataFrame(index=roi_ids, columns=['dff', 'sigma_dff', 'num_small_baseline_frames'])
    #     for i, roi_id in enumerate(roi_ids):
    #         dff_traces.loc[roi_id, 'dff'] = dff_traces_array[i, :]
    #         dff_traces.loc[roi_id, 'num_small_baseline_frames'] = num_small_baseline_frames[i]
    #         dff_traces.loc[roi_id, 'sigma_dff'] = sigma_dff[i]
    #     dff_traces.index.name = 'cell_roi_id'
    #     dff_traces = self._add_csid_to_table(dff_traces)
    #     self._dff_traces = dff_traces
    #     return self._dff_traces
    # dff_traces = LazyLoadable('_dff_traces', get_dff_traces)

    def get_dff_traces(self):

        f = h5py.File(self.file_paths['dff_h5'], mode='r')
        dff_traces_array = np.asarray(f['data'])
        #roi_ids = [int(roi_id) for roi_id in np.asarray(f['roi_names'])]
        
        # no ids in dff file, just use index
        roi_ids = np.arange(dff_traces_array.shape[0]) 
        baseline = [value for value in np.asarray(f['baseline'])]
        noise = [value for value in np.asarray(f['noise'])]
        skewness = [value for value in np.asarray(f['skewness'])]

        # convert to dataframe 
        dff_traces = pd.DataFrame(index=roi_ids, columns=['dff', 'baseline', 'noise', 'skewness'])
        for i, roi_id in enumerate(roi_ids):
            dff_traces.loc[roi_id, 'dff'] = dff_traces_array[i, :]
            dff_traces.loc[roi_id, 'baseline'] = baseline[i]
            dff_traces.loc[roi_id, 'noise'] = noise[i]
            dff_traces.loc[roi_id, 'skewness'] = skewness[i]
        dff_traces.index.name = 'cell_roi_id'
        dff_traces = self._add_csid_to_table(dff_traces)
        self._dff_traces = dff_traces
        return self._dff_traces
    dff_traces = LazyLoadable('_dff_traces', get_dff_traces)

    def get_events(self):

        f = h5py.File(self.file_paths["events_oasis_h5"], mode='r')
        events_array = np.asarray(f['events'])
        
        # just make roi_ids as for index
        #roi_ids = [int(roi_id) for roi_id in np.asarray(f['cell_roi_id'])]
        roi_ids = np.arange(events_array.shape[0])

        # convert to dataframe 
        events = pd.DataFrame(index=roi_ids, columns=['events'])
        for i, roi_id in enumerate(roi_ids):
            events.loc[roi_id, 'events'] = events_array[i, :]
        # just make nan as for each row
        events['filtered_events'] = np.nan
        events.index.name = 'cell_roi_id'
        events = self._add_csid_to_table(events)
        self._events = events
        return self._events

    def get_ophys_timestamps(self):
        sync_fp = self.file_paths['sync_file']
        ophys_timestamps = get_synchronized_frame_times(session_sync_file=sync_fp,
                                                        sync_line_label_keys=data_file_keys.OPHYS_KEYS,
                                                        drop_frames=None,
                                                        trim_after_spike=True)

        # resample for mesoscope data, planes are interleaved in sync file
        ts_len = len(ophys_timestamps)
        group_count = self.metadata['plane_group_count']
        plane_group = self.metadata['plane_group_index']
        # Sometimes, the number of timestamps across planes do not match. Force them by trimming at the end.
        trailing_frames = ts_len % group_count
        if trailing_frames != 0:
            ophys_timestamps = ophys_timestamps[:-trailing_frames]
        self._ophys_timestamps = ophys_timestamps[plane_group::group_count]
        rs_len = len(self._ophys_timestamps)

        if self.verbose:
            print(f"ophys_timestamps: {ts_len} -> {rs_len} (resampled for mesoscope data)")
        return self._ophys_timestamps

    # These data products should be available in processed data assets
    average_projection = LazyLoadable('_average_projection', get_average_projection_png)
    max_projection = LazyLoadable('_max_projection', get_max_projection_png)
    motion_transform = LazyLoadable('_motion_transform', get_motion_transform_csv)
    cell_specimen_table = LazyLoadable('_cell_specimen_table', get_cell_specimen_table)
    raw_fluorescence_traces = LazyLoadable('_raw_fluorescence_traces', get_raw_fluorescence_traces)
    neuropil_traces = LazyLoadable('_neuropil_traces', get_neuropil_traces)
    neuropil_masks = LazyLoadable('_neuropil_masks', get_neuropil_masks)
    neuropil_traces_xr = LazyLoadable('_neuropil_traces_xr', get_neuropil_traces_xr)
    demixed_traces = LazyLoadable('_demixed_traces', get_demixed_traces)
    events = LazyLoadable('_events', get_events)

    # raw/input/sessions level data products
    ophys_timestamps = LazyLoadable('_ophys_timestamps', get_ophys_timestamps)



    @classmethod
    def construct_and_load(cls, ophys_plane_id, cache_dir=None, **kwargs):
        ''' Instantiate a VisualBehaviorOphysDataset and load its data

        Parameters
        ----------
        ophys_plane_id : int
            identifier for this experiment/plane
        cache_dir : str
            directory containing this experiment/plane

        '''

        obj = cls(ophys_plane_id, cache_dir=cache_dir, **kwargs)

        obj.get_max_projection_png()
        obj.get_average_projection_png()
        obj.get_motion_transform_csv()

        # obj.get_metadata()
        # obj.get_timestamps()
        # obj.get_ophys_timestamps()
        # obj.get_stimulus_timestamps()
        # obj.get_behavior_timestamps()
        # obj.get_eye_tracking_timestamps()
        # obj.get_stimulus_presentations()
        # obj.get_stimulus_template()
        # obj.get_stimulus_metadata()
        # obj.get_running_speed()
        # obj.get_licks()
        # obj.get_rewards()
        # obj.get_task_parameters()
        # obj.get_trials()
        # obj.get_dff_traces_array()
        # obj.get_corrected_fluorescence_traces()
        # obj.get_events_array()
        # obj.get_cell_specimen_table()
        # obj.get_roi_mask_dict()
        # obj.get_roi_mask_array()
        # obj.get_cell_specimen_ids()
        # obj.get_cell_indices()
        # obj.get_dff_traces()
        # obj.get_events()
        # obj.get_pupil_area()
        # obj.get_extended_stimulus_presentations()

        return obj



# Patches
    def _patch_attributes(self):
        # Patch 1: ROI filtering
        self._filter_rois()
        pass

    def _filter_rois(self, small_roi_radius_threshold_in_um=4):
        ''' Filter ROIs based on size and motion border

        Parameters
        ----------
        small_roi_radius_threshold_in_um : float
            threshold for filtering small ROIs

        '''
        cell_specimen_table = self.cell_specimen_table
        if np.array([k in cell_specimen_table.columns for k in ['touching_motion_border', 'small_roi', 'valid_roi']]).all():
            pass
        else:
            plane_path = self.metadata['plane_path']
            range_y, range_x = lamf_utils.get_motion_correction_crop_xy_range(plane_path)
            range_y = [int(range_y[0]), int(range_y[1])]
            range_x = [int(range_x[0]), int(range_x[1])]
            
            on_mask = np.zeros((self.metadata['fov_height'], self.metadata['fov_width']), dtype=bool)
            on_mask[range_y[0]:range_y[1], range_x[0]:range_x[1]] = True
            motion_mask = ~on_mask

            def _touching_motion_border(row, motion_mask):
                if (row.mask_matrix * motion_mask).any():
                    return True
                else:
                    return False

            cell_specimen_table['touching_motion_border'] = cell_specimen_table.apply(_touching_motion_border, axis=1, motion_mask=motion_mask)
            
            small_roi_radius_threshold_in_pix = small_roi_radius_threshold_in_um / float(self.metadata['fov_scale_factor'])
            area_threshold = np.pi * (small_roi_radius_threshold_in_pix**2)
            
            cell_specimen_table['small_roi'] = cell_specimen_table['mask_matrix'].apply(lambda x: len(np.where(x)[0]) < area_threshold)
            cell_specimen_table['valid_roi'] = ~cell_specimen_table['touching_motion_border'] & ~cell_specimen_table['small_roi']
    

    
        