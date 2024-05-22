from typing import Optional

import pandas as pd
import numpy as np

from comb.core import DataObject
from comb.data_files.behavior_stimulus_file import BehaviorStimulusFile
from comb.processing.timestamps.stimulus_timestamps import StimulusTimestamps

class Rewards(DataObject):
    def __init__(self, rewards: pd.DataFrame):
        super().__init__(name='rewards', value=rewards)

    @classmethod
    def from_stimulus_file(
            cls, stimulus_file: BehaviorStimulusFile,
            stimulus_timestamps: StimulusTimestamps) -> "Rewards":
        """Get reward data from pkl file, based on timestamps
        (not sync file).
        """

        if not np.isclose(stimulus_timestamps.monitor_delay, 0.0):
            msg = ("Instantiating rewards with monitor_delay = "
                   f"{stimulus_timestamps.monitor_delay: .2e}; "
                   "monitor_delay should be zero for Rewards "
                   "data object")
            raise RuntimeError(msg)

        data = stimulus_file.data

        trial_df = pd.DataFrame(data["items"]["behavior"]["trial_log"])
        rewards_dict = {"volume": [], "timestamps": [], "auto_rewarded": []}
        for idx, trial in trial_df.iterrows():
            rewards = trial["rewards"]
            # as i write this there can only ever be one reward per trial
            if rewards:
                rewards_dict["volume"].append(rewards[0][0])
                rewards_dict["timestamps"].append(
                    stimulus_timestamps.value[rewards[0][2]])
                auto_rwrd = trial["trial_params"]["auto_reward"]
                rewards_dict["auto_rewarded"].append(auto_rwrd)

        df = pd.DataFrame(rewards_dict)
        return cls(rewards=df)