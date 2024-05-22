"""
Copied from: AllenSDK/allensdk/brain_observatory/behavior/data_objects/timestamps/stimulus_timestamps/stimulus_timestamps.py v2.16.2 03/2024
Modified by @mattjdavis

TODO:
+ sync file
"""
from typing import Optional, Union, List

import numpy as np
from pynwb import NWBFile, ProcessingModule
from pynwb.base import TimeSeries

from comb.core import DataObject
# from comb.data_files import (
#     BehaviorStimulusFile,
#     #SyncFile
# )
from comb.data_files.behavior_stimulus_file import BehaviorStimulusFile
# from comb.processing.timestamps.timestamps_processing import (
#         get_behavior_stimulus_timestamps, get_ophys_stimulus_timestamps)
from comb.processing.timestamps.timestamps_processing import get_behavior_stimulus_timestamps

class StimulusTimestamps(DataObject):
    """A DataObject which contains properties and methods to load, process,
    and represent visual behavior stimulus timestamp data.

    Stimulus timestamp data is represented as:

    Numpy array whose length is equal to the number of timestamps collected
    and whose values are timestamps (in seconds)
    """

    def __init__(
        self,
        timestamps: np.ndarray,
        monitor_delay: float,
        stimulus_file: Optional[BehaviorStimulusFile] = None,
        #sync_file: Optional[SyncFile] = None
    ):
        super().__init__(name="stimulus_timestamps",
                         value=timestamps+monitor_delay)
        self._stimulus_file = stimulus_file
        #self._sync_file = sync_file
        self._monitor_delay = monitor_delay

    def update_timestamps(
            self,
            timestamps: np.ndarray
    ) -> "StimulusTimestamps":
        """
        Returns newly instantiated `StimulusTimestamps` with `timestamps`

        Parameters
        ----------
        timestamps

        Returns
        -------
        `StimulusTimestamps` with `timestamps`
        """
        return StimulusTimestamps(
            timestamps=timestamps,
            monitor_delay=self._monitor_delay,
            stimulus_file=self._stimulus_file,
            sync_file=self._sync_file
        )

    def subtract_monitor_delay(self) -> "StimulusTimestamps":
        """
        Return a version of this StimulusTimestamps object with
        monitor_delay = 0 by subtracting self.monitor_delay from
        self.value
        """
        new_value = self.value-self.monitor_delay
        return StimulusTimestamps(
                    timestamps=new_value,
                    monitor_delay=0.0)

    @property
    def monitor_delay(self) -> float:
        return self._monitor_delay

    @classmethod
    def from_stimulus_file(
            cls,
            stimulus_file: BehaviorStimulusFile,
            monitor_delay: float) -> "StimulusTimestamps":
        stimulus_timestamps = get_behavior_stimulus_timestamps(
            stimulus_pkl=stimulus_file.data
        )

        return cls(
            timestamps=stimulus_timestamps,
            monitor_delay=monitor_delay,
            stimulus_file=stimulus_file
        )

    # @classmethod
    # def from_sync_file(
    #         cls,
    #         sync_file: SyncFile,
    #         monitor_delay: float) -> "StimulusTimestamps":
    #     stimulus_timestamps = get_ophys_stimulus_timestamps(
    #         sync_path=sync_file.filepath
    #     )
    #     return cls(
    #         timestamps=stimulus_timestamps,
    #         monitor_delay=monitor_delay,
    #         sync_file=sync_file
    #     )