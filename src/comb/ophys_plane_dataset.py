from comb.ophys_plane_grabber import OphysPlaneGrabber
from comb.processing.sync.sync_utilities import get_synchronized_frame_times

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
                raw_folder_path: Optional[str] = None, # where sync file is (pkl file)
                opid: Optional[str] = None,
                data_path: Optional[str] = None,
                verbose=False):
        super().__init__(plane_folder_path=plane_folder_path,
                         raw_folder_path=raw_folder_path,
                         opid=opid,
                         data_path=data_path,
                         verbose=verbose)

        self.metadata = self._set_metadata()

        # keep for legacy purposes
        self.ophys_experiment_id = self._resolve_ophys_experiment_id()

        # no opid is given, set as= ophys_experiment_id
        if self.opid is None:
            self.opid = self.ophys_experiment_id


        # currently pipeline give all nan traces; lets remove
        try:
            self._set_all_nan_traces_invalid()
        except TypeError: # No dff_file
            pass

        self.metadata['ophys_frame_rate'] = self._get_ophys_frame_rate()


    ####################################################################
    # Data files
    ####################################################################


    def _resolve_ophys_experiment_id(self):
        
        if self.plane_folder_path is not None:
            ophys_experiment_id = self.plane_folder_path.stem
        elif self.opid is not None:
            ophys_experiment_id = self.opid

        return ophys_experiment_id


    def _parse_mesoscope_metadata(self):
        # assert self.file_paths['mesoscope_splitting_json'] not none
        assert self.file_paths['mesoscope_splitting_json'] is not None, "mesoscope_splitting_json is not found, only mesoscope data is supported"
        
        split_dict = {}
        with open(self.file_paths['mesoscope_splitting_json']) as json_file:
            split_json = json.load(json_file)

        split_dict['plane_group_count'] = len(split_json['plane_groups'])

        for i, plane_group in enumerate(split_json['plane_groups']):
            for plane_dict in plane_group['ophys_experiments']:
                # find index of plane['experiment_id'] that matches self.opid
                if str(plane_dict['experiment_id']) == self.opid:
                    split_dict['roi_index'] = plane_dict['roi_index']
                    split_dict['plane_group_index'] = i
                    split_dict['scanfield_z'] = plane_dict['scanfield_z'] # TODO rename 

        return split_dict

    def _get_ophys_frame_rate(self):

        dt = self.ophys_timestamps.diff().mean()
        frame_rate = 1/dt

        if self.verbose:
            print("Calculating frame rate from ophys_timestamps, not metadata")

        return frame_rate



    def _set_metadata(self):
        metadata = {}
        with open(self.file_paths['platform_json']) as json_file:
            platform = json.load(json_file)

        split_dict = self._parse_mesoscope_metadata()
        metadata.update(split_dict)

        return metadata

    def _set_all_nan_traces_invalid(self):

        dff = self.dff_traces
        nan_ids = []
        # iterate dff.dff, check if array is all nan
        for cell_specimen_id, trace in dff.dff.items():
            if np.all(np.isnan(trace)):
                nan_ids.append(cell_specimen_id)
        print(nan_ids)

        new_csid_table = self.cell_specimen_table
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
        self._motion_transform = pd.read_csv()
        return self._motion_transform

    # TODO: should we rename the attribute to segmentation? (MJD)
    def get_cell_specimen_table(self): 
        with open(self.file_paths['segmentation_output_json']) as json_file:
            segmentation_output = json.load(json_file)
        cell_specimen_table = pd.DataFrame(segmentation_output)
        cell_specimen_table = cell_specimen_table.rename(columns={'id': 'cell_roi_id'})
        print(cell_specimen_table.columns)
        cell_specimen_table = self._add_csid_to_table(cell_specimen_table)
        self._cell_specimen_table = cell_specimen_table
        return self._cell_specimen_table

    def get_raw_fluorescence_traces(self):

        with h5py.File(self.file_paths['roi_traces_h5'], 'r') as f:
            traces = np.asarray(f['data'])
            roi_ids = [int(roi_id) for roi_id in np.asarray(f['roi_names'])]

        traces_df = pd.DataFrame(index=roi_ids, columns=['raw_fluorescence_traces'])
        for i, roi_id in enumerate(roi_ids):
            traces_df.loc[roi_id, 'raw_fluorescence_traces'] = traces[i, :]
        traces_df = traces_df.rename(columns={'roi_id': 'cell_roi_id'})
#        traces_df = self._add_csid_to_table(traces_df)
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
        roi_ids = [int(roi_id) for roi_id in np.asarray(f['roi_names'])]
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
        roi_ids = [int(roi_id) for roi_id in np.asarray(f['cell_roi_id'])]

        # convert to dataframe 
        events = pd.DataFrame(index=roi_ids, columns=['events'])
        for i, roi_id in enumerate(roi_ids):
            events.loc[roi_id, 'events'] = events_array[i, :]
        events['filtered_events'] = events['events']
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