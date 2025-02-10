from comb.behavior_session_grabber import BehaviorSessionGrabber

from comb.processing.stimulus import stimulus_processing
from comb.processing.biometrics import running_processing
from comb.processing.sync import sync_utilities
from comb.processing.sync import time_sync
from comb.processing.biometrics.licks import Licks

from comb.data_files.behavior_stimulus_file import BehaviorStimulusFile

from comb.processing.timestamps.stimulus_timestamps import StimulusTimestamps
from comb.processing.stimulus.presentations import Presentations
from comb.processing.biometrics.rewards import Rewards

import comb.processing.eye_tracking as eye_tracking
from comb.processing.eye_tracking_table import EyeTrackingTable
from comb.data_files.eye_tracking_file import EyeTrackingFile

from . import data_file_keys

from typing import Any, Optional, Union
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import h5py
import numpy as np
import xarray as xr
from scipy import ndimage
from pathlib import Path

from pathlib import Path

import logging
logger = logging.getLogger(__name__)


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


class BehaviorSessionDataset(BehaviorSessionGrabber):
    """Includes stimulus, tasks, biometrics"""
    def __init__(self, 
                 raw_folder_path: Union[str, Path] = None, # where sync file is (pkl file)
                 oeid: Optional[str] = None,
                 data_path: Optional[str] = None,
                 monitor_delay: float = 0.0,
                 eye_tracking_path: Optional[Union[str, Path]] = None,
                 project_code: Optional[str] = None,
                 verbose: Optional[bool] = False,
                 apply_patch: Optional[bool] = True):
        super().__init__(raw_folder_path=raw_folder_path,
                         oeid=oeid,
                         data_path=data_path)

        self.verbose = verbose
        self._load_behavior_stimulus_file()
        self.monitor_delay = monitor_delay # TODO: UPDATE
        self.project_code = project_code
        self.get_stimulus_timestamps()

        self.raw_folder_path = Path(raw_folder_path)
        self.session_name = self.raw_folder_path.name

        if eye_tracking_path is not None:
            self.file_paths['eye_tracking'] = Path(eye_tracking_path)
            #self.eye_tracking = _load_and_process_eye_tracking()
        
        self.metadata = None
        # TODO metadata

        # Patch some of the issues, until they are resolved.
        # 1. Adding trials information - dev on the way
        # 2. Remove pupil area outliers - issue documented. Need to resolve from ellipse calculation (pupil dlc capsule)
        if apply_patch:
            self._patch_attributes()

    def _load_behavior_stimulus_file(self):
        # load file when BehaviorDataset is instantiated
        self.behavior_stimulus_file = BehaviorStimulusFile.from_file(self.file_paths['stimulus_pkl'])

    def get_stimulus_timestamps(self): 
        self._stimulus_timestamps = sync_utilities.get_synchronized_frame_times(
            session_sync_file=self.file_paths['sync_file'],
            sync_line_label_keys=data_file_keys.STIMULUS_KEYS,
            drop_frames=None,
            trim_after_spike=True)

        return self._stimulus_timestamps

    # def get_stimulus_presentations_old(self):
    #     """"This is an old method; skips Presentations class, which has more updated processing"""
    #     pkl_file_path = self.file_paths['stimulus_pkl']
    #     pkl_data = pd.read_pickle(pkl_file_path)

    #     self._stimulus_presentations = stimulus_processing.get_stimulus_presentations(
    #         pkl_data, self.stimulus_timestamps)

    #     return self._stimulus_presentations

    # def get_eye_tracking_table_old(self):
    #     """Load and process eye tracking data,
    #     this is an old method; just directly load the table, without adding timestamps"""

    #     try:
    #         eye_tracking_file = self.file_paths['eye_tracking'] / "ellipses_processed.h5"
    #         eye_tracking_table = eye_tracking.load_eye_tracking_hdf(eye_tracking_file)

    #         # todo: maybe create additional processing (handle metadata frame + stimulus timestamps)
    #         #processed_eye_tracking_data = process_eye_tracking_data(eye_tracking_data)
    #     except:
    #         eye_tracking_table = None

    #     self._eye_tracking_table = eye_tracking_table
    
    #     return self._eye_tracking_table
    # eye_tracking = LazyLoadable('_eye_tracking_table', get_eye_tracking_table)
    
    
    def get_eye_tracking_table(self):
        """Load and process eye tracking data"""
        verbose = self.verbose
        try:
            eye_tracking_path = self.file_paths['eye_tracking'] / "ellipses_processed.h5"
            eye_tracking_df = EyeTrackingFile.load_data(filepath=eye_tracking_path)
            
            frame_times = sync_utilities.get_synchronized_frame_times(
                session_sync_file=self.file_paths['sync_file'],
                sync_line_label_keys=data_file_keys.EYE_TRACKING_KEYS,
                drop_frames=None,
                trim_after_spike=False)
            
            stimulus_timestamps = StimulusTimestamps(
                timestamps=frame_times.to_numpy(),
                monitor_delay=0.0)
            
            eye_tracking_table = EyeTrackingTable.from_data_file(data_file=eye_tracking_df, 
                                                                 stimulus_timestamps=stimulus_timestamps)
            if verbose:
                logger.info("Loaded eye tracking data from: " + str(eye_tracking_path))
        except Exception as e:            
            if 'eye_tracking' not in self.file_paths.keys():
                logger.error("eye_tracking not defined as a file_path", exc_info=True)
            else:
                eye_tracking_path = self.file_paths['eye_tracking'] / "ellipses_processed.h5"
                logger.error("Could not load eye tracking data from: " + str(eye_tracking_path), exc_info=True)
            eye_tracking_table = None

        self._eye_tracking_table = eye_tracking_table
    
        return self._eye_tracking_table
    eye_tracking_table = LazyLoadable('_eye_tracking_table', get_eye_tracking_table)

    def get_stimulus_presentations(self, monitor_delay=0.03613):
        """"TODO"""
        # alternative timestamps, from pkl file. produces 
        # stimulus_timestamps = StimulusTimestamps.from_stimulus_file(self.behavior_stimulus_file, 
        #                                            monitor_delay=self.monitor_delay)

        # Calculate monitor delay
        # Only for STAGE_1 for now
        # TODO: Add for other session types, and make it generalized (At least within certain scope of stimuli sets)
        # session_type = self.behavior_dataset.behavior_stimulus_file.session_type
        session_type = self.behavior_stimulus_file.session_type
        # if session_type == 'STAGE_1':
        #     monitor_delay = self.get_monitor_delay_stage_1()

        stimulus_timestamps = StimulusTimestamps(timestamps=self.stimulus_timestamps, monitor_delay=monitor_delay)
        
        # if session_type == 'STAGE_1':
        #     self._stimulus_presentations = self.get_stage_1_stimulus_presentations(stimulus_timestamps=stimulus_timestamps)
        # else:
        st = Presentations.from_stimulus_file(stimulus_file=self.behavior_stimulus_file,
                                                stimulus_timestamps=stimulus_timestamps,
                                                project_code=self.project_code) # TODO: GET BEHAVIOR SESSION ID

        self._stimulus_presentations = st.value # TODO: probably smoother way to return than call value

        return self._stimulus_presentations


    def get_monitor_delay_stage_1(self, verbose=True):
        ''' Get monitor delay for STAGE_1 session type'''
        assert self.behavior_stimulus_file.session_type == 'STAGE_1'

        sync_filepath = self.file_paths['sync_file']
        hdf = h5py.File(sync_filepath, "r")
        assert list(hdf.keys()) == ['data', 'meta']

        # Read sync data
        data = hdf['data'][:]
        sync_sig = data[:, -1]
        time_stamps = data[:, 0]

        # Read metadata
        meta_str = hdf['meta'][()].decode('UTF-8').replace('\'','\"')
        meta = json.loads(meta_str)
        line_labels = meta['line_labels']
        sample_freq = meta['ni_daq']['sample_rate']

        # Finding rising and falling edges
        # TODO: probably calculate it from a separate file...? (there was one in the allenSDK)
        falling_edges = {}
        rising_edges = {}
        
        for bit, label in enumerate(line_labels):
            if not label:
                label = 'sig_' + str(bit)
        
            bit_array = np.bitwise_and(sync_sig, 2 ** bit).astype(bool).astype(np.uint8)
            bit_changes = np.ediff1d(bit_array, to_begin=0)
            falling_edges[label] = time_stamps[np.where(bit_changes == 255)]/sample_freq
            rising_edges[label] = time_stamps[np.where(bit_changes == 1)]/sample_freq

        delay_mu, delay_sd = time_sync.calculate_monitor_delay_visual_coding(rising_edges['stim_photodiode'], falling_edges['vsync_stim'])

        if verbose:
            print("monitor delay: " + str(delay_mu) + '+,-' + str(delay_sd))

        return delay_mu

    # This is hard to do without the SDK
    # def get_stimulus_template(self):
    #     return self._stimulus_template
    # stimulus_template = LazyLoadable('_stimulus_template', get_stimulus_template)

     # This is hard to do without the SDK, maybe?
    # def get_stimulus_metadata(self, append_omitted_to_stim_metadata=True):
    #     return self._stimulus_metadata
    # stimulus_metadata = LazyLoadable('_stimulus_metadata', get_stimulus_metadata)


    def get_running_speed(self):
        zscore_threshold = 10.0
        lowpass_filter = True

        stimulus_timestamps = self.stimulus_timestamps 

        running_data_df = running_processing.get_running_df(
            data=self.behavior_stimulus_file.data, time=stimulus_timestamps,
            lowpass=lowpass_filter, zscore_threshold=zscore_threshold)

        self._running_speed = pd.DataFrame({
            "timestamps": running_data_df.index.values,
            "speed": running_data_df.speed.values})
        return self._running_speed
    running_speed = LazyLoadable('_running_speed', get_running_speed)


    def get_licks(self):
        self._licks = Licks.from_stimulus_file(
            stimulus_file=self.behavior_stimulus_file,
            stimulus_timestamps=self.stimulus_timestamps)
        return self._licks
    licks = LazyLoadable('_licks', get_licks)


    def get_rewards(self):
        st = StimulusTimestamps.from_stimulus_file(self.behavior_stimulus_file, 
                                                   monitor_delay=self.monitor_delay)

        self._rewards = Rewards.from_stimulus_file(
                stimulus_file=self.behavior_stimulus_file,
                stimulus_timestamps=st.subtract_monitor_delay()).value # TODO: value?
        return self._rewards
    rewards = LazyLoadable('_rewards', get_rewards)


    # def get_task_parameters(self):
    #     self._task_parameters = 
    #     return self._task_parameters
    # task_parameters = LazyLoadable('_task_parameters', get_task_parameters)


    # def get_trials(self):
        #   self._trials = 
    #     return self._trials
    # trials = LazyLoadable('_trials', get_trials)

    # lazy load
    stimulus_presentations = LazyLoadable('_stimulus_presentations', get_stimulus_presentations)
    stimulus_timestamps = LazyLoadable('_stimulus_timestamps', get_stimulus_timestamps)

    # @classmethod
    # def _read_behavior_stimulus_timestamps(
    #     cls,
    #     stimulus_file_lookup: StimulusFileLookup,
    #     sync_file: Optional[SyncFile],
    #     monitor_delay: float,
    # ) -> StimulusTimestamps:
    #     """
    #     Assemble the StimulusTimestamps from the SyncFile.
    #     If a SyncFile is not available, use the
    #     behavior_stimulus_file
    #     """
    #     if sync_file is not None:
    #         stimulus_timestamps = StimulusTimestamps.from_sync_file(
    #             sync_file=sync_file, monitor_delay=monitor_delay
    #         )
    #     else:
    #         stimulus_timestamps = StimulusTimestamps.from_stimulus_file(
    #             stimulus_file=stimulus_file_lookup.behavior_stimulus_file,
    #             monitor_delay=monitor_delay,
    #         )

    #     return stimulus_timestamps


    # @classmethod
    # def _read_licks(
    #     cls,
    #     behavior_stimulus_file: BehaviorStimulusFile,
    #     sync_file: Optional[SyncFile],
    #     monitor_delay: float,
    # ) -> Licks:
    #     """
    #     Construct the Licks data object for this session

    #     Note: monitor_delay is a part of the call signature so that
    #     it can be used in sub-class implementations of this method.
    #     """

    #     stimulus_timestamps = cls._read_behavior_stimulus_timestamps(
    #         sync_file=sync_file,
    #         stimulus_file_lookup=stimulus_file_lookup,
    #         monitor_delay=0.0,
    #     )

    #     return Licks.from_stimulus_file(
    #         stimulus_file=stimulus_file_lookup.behavior_stimulus_file,
    #         stimulus_timestamps=stimulus_timestamps,
    #     )

    # @classmethod
    # def _read_rewards(
    #     cls,
    #     stimulus_file_lookup: StimulusFileLookup,
    #     sync_file: Optional[SyncFile],
    # ) -> Rewards:
    #     """
    #     Construct the Rewards data object for this session
    #     """
    #     stimulus_timestamps = cls._read_behavior_stimulus_timestamps(
    #         sync_file=sync_file,
    #         stimulus_file_lookup=stimulus_file_lookup,
    #         monitor_delay=0.0,
    #     )

    #     return Rewards.from_stimulus_file(
    #         stimulus_file=stimulus_file_lookup.behavior_stimulus_file,
    #         stimulus_timestamps=stimulus_timestamps.subtract_monitor_delay(),
    #     )


    # @classmethod
    # def _read_data_from_stimulus_file(
    #     cls,
    #     stimulus_file_lookup: StimulusFileLookup,
    #     behavior_session_id: int,
    #     sync_file: Optional[SyncFile],
    #     monitor_delay: float,
    #     include_stimuli: bool = True,
    #     stimulus_presentation_columns: Optional[List[str]] = None,
    #     project_code: Optional[ProjectCode] = None,
    # ):
    #     """Helper method to read data from stimulus file"""

    #     licks = cls._read_licks(
    #         stimulus_file_lookup=stimulus_file_lookup,
    #         sync_file=sync_file,
    #         monitor_delay=monitor_delay,
    #     )

    #     rewards = cls._read_rewards(
    #         stimulus_file_lookup=stimulus_file_lookup, sync_file=sync_file
    #     )

    #     session_stimulus_timestamps = cls._read_session_timestamps(
    #         stimulus_file_lookup=stimulus_file_lookup,
    #         sync_file=sync_file,
    #         monitor_delay=monitor_delay,
    #     )

    #     trials = cls._read_trials(
    #         stimulus_file_lookup=stimulus_file_lookup,
    #         sync_file=sync_file,
    #         monitor_delay=monitor_delay,
    #         licks=licks,
    #         rewards=rewards,
    #     )

    #     if include_stimuli:
    #         stimuli = cls._read_stimuli(
    #             stimulus_file_lookup=stimulus_file_lookup,
    #             behavior_session_id=behavior_session_id,
    #             sync_file=sync_file,
    #             monitor_delay=monitor_delay,
    #             trials=trials,
    #             stimulus_presentation_columns=stimulus_presentation_columns,
    #             project_code=project_code,
    #         )
    #     else:
    #         stimuli = None

    #     task_parameters = TaskParameters.from_stimulus_file(
    #         stimulus_file=stimulus_file_lookup.behavior_stimulus_file
    #     )

    #     return (
    #         session_stimulus_timestamps.subtract_monitor_delay(),
    #         licks,
    #         rewards,
    #         stimuli,
    #         task_parameters,
    #         trials,
    #     )


    # Patches
    def _patch_attributes(self):
        # Patch 1: Add trials information
        self._add_trials_info()
        # Patch 2: Remove pupil area outliers
        self._remove_pupil_area_outliers()

    
    def _add_trials_info(self, response_window=(0.15, 0.75)):
        """ Temporary fix to add trials to bod
        using stimulus_presentations and licks.
        No correct rejection for this.
        Columns: 'change_time', 'hit', 'miss'

        Parameters
        ----------
        bod : BehaviorOphysDataset
            The behavior ophys dataset object.

        Returns
        -------
        bod : BehaviorOphysDataset
            The behavior ophys dataset object with trials.
        """

        stimulus_presentations = self.stimulus_presentations
        lick_times = self.licks.timestamps.values
        trials = pd.DataFrame(columns=['change_time', 'hit', 'miss'])

        stimulus_presentations['is_change'] = stimulus_presentations['is_change'].astype('boolean').fillna(False).astype(bool)
        change_times = stimulus_presentations.query('is_change').start_time.values
        response_windows = np.array([change_times + response_window[0], change_times + response_window[1]]).T
        hit = np.zeros(len(change_times), 'bool')
        for i, window in enumerate(response_windows):
            if np.any((lick_times > window[0]) & (lick_times < window[1])):
                hit[i] = 1
        miss = ~hit
        trials = pd.DataFrame({'change_time': change_times, 'hit': hit, 'miss': miss})

        self.trials = trials
    

    def _remove_pupil_area_outliers(self, tick_threshold=None,
                                    tick_std_multiplier_threshold=20,
                                    dilation_frames: int = 2,
                                    pupil_average_confidence_threshold: float = 0.65):
        """
        Remove pupil area outliers from eye_tracking_table.
            Remove frames that change more than tick_treshold values or more than tick_std_multiplier_threshold stds
            If both are None, no frames are removed.
            If both are not None, tick_threshold is used.
            All threshold is on the absolute frame-to-frame difference.
            Filtering is based on pupil area only.
        Also add a mask for low confidence frames (average likelihood of pupil points).
        Results applied to all.
        """
        eye_df = self.eye_tracking_table

        if tick_threshold is None:
            if tick_std_multiplier_threshold is not None:
                tick_threshold = tick_std_multiplier_threshold * eye_df['pupil_area'].diff().abs().std()
        
        outlier_mask = np.zeros(len(eye_df), 'bool')
        if tick_threshold is not None:
            # Get pupil area change rate outlier mask
            outlier_mask = eye_df.pupil_area.diff().abs().fillna(0) > tick_threshold

        if pupil_average_confidence_threshold is not None:
            # Add low confidence mask
            low_confidence_mask = eye_df['pupil_average_confidence'].values < pupil_average_confidence_threshold
            outlier_mask = outlier_mask | low_confidence_mask

        if dilation_frames > 0:
            outlier_mask = ndimage.binary_dilation(outlier_mask,
                                                    iterations=dilation_frames)
    
        eye_df = eye_df[~outlier_mask]
        self.eye_tracking_table = eye_df

