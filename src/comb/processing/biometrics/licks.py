from comb.data_files.behavior_stimulus_file import BehaviorStimulusFile

from typing import Optional, Union
import numpy as np
import pandas as pd

class Licks(object):

    def __init__(self, licks: pd.DataFrame):
        """
        :param licks
            dataframe containing the following columns:
                - timestamps: float
                    stimulus timestamps in which there was a lick
                - frame: int
                    frame number in which there was a lick
        """
        self.data = licks

    def __getattr__(self, attr):
        # If the attribute being accessed is not found in the instance, try to get it from the DataFrame
        if hasattr(self.data, attr):
            return getattr(self.data, attr)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")
    
    @classmethod
    def from_stimulus_file(
            cls,
            stimulus_file: BehaviorStimulusFile,
            #stimulus_timestamps: Union[StimulusTimestamps, np.ndarray]
            stimulus_timestamps: np.ndarray
            ) -> "Licks":
        """Get lick data from pkl file.
        This function assumes that the first sensor in the list of
        lick_sensors is the desired lick sensor.

        Since licks can occur outside of a trial context, the lick times
        are extracted from the vsyncs and the frame number in `lick_events`.
        Since we don't have a timestamp for when in "experiment time" the
        vsync stream starts (from self.get_stimulus_timestamps), we compute
        it by fitting a linear regression (frame number x time) for the
        `start_trial` and `end_trial` events in the `trial_log`, to true
        up these time streams.

        Parameters
        ----------
        stimulus_file : BehaviorStimulusFile
            Input Behavior stims loaded from a pickle file.
        stimulus_timestamps : StimulusTimestamps or np.ndarray
            Timestamps containing lick data either in a StimulusTimestamps
            object or numpy array. Numpy array data must be the SyncFile
            line named ``lick_times``.

        Returns
        -------
        `Licks` instance
        """
        data = stimulus_file.data

        lick_frames = (data["items"]["behavior"]["lick_sensors"][0]
                       ["lick_events"])

        # MJD remove
        # if isinstance(stimulus_timestamps, StimulusTimestamps):
        #     if not np.isclose(stimulus_timestamps.monitor_delay, 0.0):
        #         msg = ("Instantiating licks with monitor_delay = "
        #                f"{stimulus_timestamps.monitor_delay: .2e}; "
        #                "monitor_delay should be zero for Licks "
        #                "data object")
        #         raise RuntimeError(msg)

        #     lick_times = stimulus_timestamps.value
        # else:
        #     lick_times = stimulus_timestamps

        lick_times = stimulus_timestamps

        # there's an occasional bug where the number of logged
        # frames is one greater than the number of vsync intervals.
        # If the animal licked on this last frame it will cause an
        # error here. This fixes the problem.
        # see: https://github.com/AllenInstitute/visual_behavior_analysis
        # /issues/572  # noqa: E501
        #    & https://github.com/AllenInstitute/visual_behavior_analysis
        #    /issues/379  # noqa:E501
        #
        # This bugfix copied from
        # https://github.com/AllenInstitute/visual_behavior_analysis/blob

        if len(lick_frames) > 0:
            if lick_frames[-1] == len(lick_times):
                lick_frames = lick_frames[:-1]
                # cls._logger.error('removed last lick - '
                #                   'it fell outside of stimulus_timestamps '
                #                   'range')
        # # MJD REMOVE
        # if isinstance(stimulus_timestamps, StimulusTimestamps):
        #     lick_times = np.array([lick_times[frame] for frame in lick_frames])

        # Since we are giving general stim ts, need to get the correct times, just like above
        lick_times = np.array([lick_times[frame] for frame in lick_frames])

        # Make sure licks are the same length as number of frames (mostly for
        # array input).
        max_length = min(len(lick_times), len(lick_frames))
        lick_frames = lick_frames[0:max_length]
        lick_times = lick_times[0:max_length]

        df = pd.DataFrame({"timestamps": lick_times, "frame": lick_frames})
        return cls(licks=df)
