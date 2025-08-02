from comb.data_files.behavior_stimulus_file import BehaviorStimulusFile

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def one(x):
    if isinstance(x, str):
        return x
    try:
        xlen = len(x)
    except TypeError:
        return x
    if xlen != 1:
        raise OneResultExpectedError("Expected length one result, received: "
                                    f"{x} results from query")
    if isinstance(x, set):
        return list(x)[0]
    else:
        return x[0]


def get_trial_bounds(trial_log: list):
    """
    Adjust trial boundaries from a trial_log so that there is no dead time
    between trials.

    Parameters
    ----------
    trial_log: list
        The trial_log read in from the well known behavior stimulus
        pickle file

    Returns
    -------
    list
        Each element in the list is a tuple of the form
        (start_frame, end_frame) so that the ith element
        of the list gives the start and end frames of
        the ith trial. The endframe of the last trial will
        be -1, indicating that it should map to the last
        timestamp in the session
    """
    start_frames = []

    for trial in trial_log:
        start_f = None
        for event in trial["events"]:
            if event[0] == "trial_start":
                start_f = event[-1]
                break
        if start_f is None:
            msg = "Could not find a 'trial_start' event "
            msg += "for all trials in the trial log\n"
            msg += f"{trial}"
            raise ValueError(msg)

        if len(start_frames) > 0 and start_f < start_frames[-1]:
            msg = "'trial_start' frames in trial log "
            msg += "are not in ascending order"
            msg += f"\ntrial_log: {trial_log}"
            raise ValueError(msg)

        start_frames.append(start_f)

    end_frames = [idx for idx in start_frames[1:] + [-1]]
    return list([(s, e) for s, e in zip(start_frames, end_frames)])


def columns_to_output():
    """
    Return the list of columns in the correct order for trials table
    """
    return [
        "initial_image_name",
        "change_image_name",
        "stimulus_change",
        "change_time",
        "go",
        "catch",
        "lick_times",
        "response_time",
        "response_latency",
        "reward_time",
        "reward_volume",
        "hit",
        "false_alarm",
        "miss",
        "correct_reject",
        "aborted",
        "auto_rewarded",
        "change_frame",
        "start_time",
        "stop_time",
        "trial_length",
    ]

class Trial:
    def __init__(
            self,
            trial: dict,
            start: float,
            end: float,
            pkl_data: dict,
            index: int,
            stimulus_timestamps: list,
            licks: pd.DataFrame,
            rewards: pd.DataFrame,
            stimuli: dict,
            sync_file: Optional):
        """
        sync_file is an argument that will be used by
        sub-classes that have a more subtle way of handling
        monitor delay.
        """

        self._trial = trial
        self._start = start
        self._end = self._calculate_trial_end(
            trial_end=end, pkl_data=pkl_data)
        self._index = index
        self._stimulus_timestamps = stimulus_timestamps
        self._sync_file = sync_file
        self._data = self._match_to_sync_timestamps(
            raw_stimulus_timestamps=stimulus_timestamps,
            licks=licks,
            rewards=rewards,
            stimuli=stimuli)

    @property
    def data(self):
        return self._data

    def _match_to_sync_timestamps(
            self,
            raw_stimulus_timestamps: pd.Series,
            licks: object,
            rewards: object,
            stimuli: dict):
        """
        raw_stimulus_timestamps include monitor_delay
        """

        # Need to separate out the monitor_delay from the un-corrected timestamps
        # get version of timestamps with monitor_delay = 0 by subtracting it
        monitor_delay = 0.02115
        # This is the median estimate across all rigs
        # as discussed in https://github.com/AllenInstitute/AllenSDK/issues/1318
        stimulus_timestamps = raw_stimulus_timestamps.values-monitor_delay

        event_dict = {
            (e[0], e[1]): {
                'timestamp': stimulus_timestamps[e[3]],
                'frame': e[3]} for e in self._trial['events']
        }

        tr_data = {"trial": self._trial["index"]}
        lick_frames = licks.data['frame'].values
        timestamps = stimulus_timestamps
        reward_times = rewards['timestamps'].values

        # this block of code is trying to mimic
        # https://github.com/AllenInstitute/visual_behavior_analysis
        # /blob/master/visual_behavior/translator/foraging2
        # /stimulus_processing.py
        # #L377-L381
        # https://github.com/AllenInstitute/visual_behavior_analysis
        # /blob/master/visual_behavior/translator/foraging2
        # /extract_movies.py#L59-L94
        # https://github.com/AllenInstitute/visual_behavior_analysis
        # /blob/master/visual_behavior/translator/core/annotate.py#L11-L36
        #
        # In summary: there are cases where an "epilogue movie" is shown
        # after the proper stimuli; we do not want licks that occur
        # during this epilogue movie to be counted as belonging to
        # the last trial
        # https://github.com/AllenInstitute/visual_behavior_analysis
        # /issues/482

        # select licks that fall between trial_start and trial_end;
        # licks on the boundary get assigned to the trial that is ending,
        # rather than the trial that is starting
        if self._end > 0:
            valid_idx = np.where(np.logical_and(lick_frames > self._start,
                                                lick_frames <= self._end))
        else:
            valid_idx = np.where(lick_frames > self._start)

        valid_licks = lick_frames[valid_idx]
        if len(valid_licks) > 0:
            tr_data["lick_times"] = timestamps[valid_licks]
        else:
            tr_data["lick_times"] = np.array([], dtype=float)

        tr_data["reward_time"] = self._get_reward_time(
            reward_times,
            event_dict[('trial_start', '')]['timestamp'],
            event_dict[('trial_end', '')]['timestamp']
        )
        tr_data.update(self._get_trial_data())
        tr_data.update(self._get_trial_timing(
            event_dict,
            tr_data['lick_times'],
            tr_data['go'],
            tr_data['catch'],
            tr_data['auto_rewarded'],
            tr_data['hit'],
            tr_data['false_alarm'],
            tr_data["aborted"],
        ))
        tr_data.update(self._get_trial_image_names(stimuli))

        self._validate_trial_condition_exclusivity(tr_data=tr_data)

        return tr_data


    @staticmethod
    def _get_reward_time(rebased_reward_times,
                         start_time,
                         stop_time) -> float:
        """extract reward times in time range"""
        reward_times = rebased_reward_times[np.where(np.logical_and(
            rebased_reward_times >= start_time,
            rebased_reward_times <= stop_time
        ))]
        return float('nan') if len(reward_times) == 0 else one(
            reward_times)

    @staticmethod
    def _calculate_trial_end(
            trial_end,
            pkl_data) -> int:
        if trial_end < 0:
            bhv = pkl_data['items']['behavior']['items']
            if 'fingerprint' in bhv.keys():
                trial_end = bhv['fingerprint']['starting_frame']
        return trial_end

    def _get_trial_data(self):
        """
        Infer trial logic from trial log. Returns a dictionary.

        * reward volume: volume of water delivered on the trial, in mL

        Each of the following values is boolean:

        Trial category values are mutually exclusive
        * go: trial was a go trial (trial with a stimulus change)
        * catch: trial was a catch trial (trial with a sham stimulus change)

        stimulus_change/sham_change are mutually exclusive
        * stimulus_change: did the stimulus change (True on 'go' trials)
        * sham_change: stimulus did not change, but response was evaluated
                       (True on 'catch' trials)

        Each trial can be one (and only one) of the following:
        * hit (stimulus changed, animal responded in response window)
        * miss (stimulus changed, animal did not respond in response window)
        * false_alarm (stimulus did not change,
                       animal responded in response window)
        * correct_reject (stimulus did not change,
                          animal did not respond in response window)
        * aborted (animal responded before change time)
        * auto_rewarded (reward was automatically delivered following the
        change.
                         This will bias the animals choice and should not be
                         categorized as hit/miss)
        """
        trial_event_names = [val[0] for val in self._trial['events']]
        hit = 'hit' in trial_event_names
        false_alarm = 'false_alarm' in trial_event_names
        miss = 'miss' in trial_event_names
        sham_change = 'sham_change' in trial_event_names
        stimulus_change = 'stimulus_changed' in trial_event_names
        aborted = 'abort' in trial_event_names

        if aborted:
            go = catch = auto_rewarded = False
        else:
            catch = self._trial["trial_params"]["catch"] is True
            auto_rewarded = self._trial["trial_params"]["auto_reward"]
            go = not catch and not auto_rewarded

        correct_reject = catch and not false_alarm

        if auto_rewarded:
            hit = miss = correct_reject = false_alarm = False

        return {
            "reward_volume": sum([
                r[0] for r in self._trial.get("rewards", [])]),
            "hit": hit,
            "false_alarm": false_alarm,
            "miss": miss,
            "sham_change": sham_change,
            "stimulus_change": stimulus_change,
            "aborted": aborted,
            "go": go,
            "catch": catch,
            "auto_rewarded": auto_rewarded,
            "correct_reject": correct_reject,
        }

    def _get_trial_timing(
            self,
            event_dict: dict,
            licks: List[float],
            go: bool,
            catch: bool,
            auto_rewarded: bool,
            hit: bool,
            false_alarm: bool,
            aborted: bool):
        """
        Extract a dictionary of trial timing data.
        See trial_data_from_log for a description of the trial types.

        Parameters
        ==========
        event_dict: dict
            Dictionary of trial events in the well-known `pkl` file
        licks: List[float]
            list of lick timestamps, from the `get_licks` response for
            the BehaviorOphysExperiment.api.
        go: bool
            True if "go" trial, False otherwise. Mutually exclusive with
            `catch`.
        catch: bool
            True if "catch" trial, False otherwise. Mutually exclusive
            with `go.`
        auto_rewarded: bool
            True if "auto_rewarded" trial, False otherwise.
        hit: bool
            True if "hit" trial, False otherwise
        false_alarm: bool
            True if "false_alarm" trial, False otherwise
        aborted: bool
            True if "aborted" trial, False otherwise

        Returns
        =======
        dict
            start_time: float
                The time the trial started (in seconds elapsed from
                recording start)
            stop_time: float
                The time the trial ended (in seconds elapsed from
                recording start)
            trial_length: float
                Duration of the trial in seconds
            response_time: float
                The response time, for non-aborted trials. This is equal
                to the first lick in the trial. For aborted trials or trials
                without licks, `response_time` is NaN.
            change_frame: int
                The frame number that the stimulus changed
            change_time: float
                The time in seconds that the stimulus changed
            response_latency: float or None
                The time in seconds between the stimulus change and the
                animal's lick response, if the trial is a "go", "catch", or
                "auto_rewarded" type. If the animal did not respond,
                return `float("inf")`. In all other cases, return None.

        Notes
        =====
        The following parameters are mutually exclusive (exactly one can
        be true):
            hit, miss, false_alarm, aborted, auto_rewarded
        """
        assert not (aborted and (hit or false_alarm or auto_rewarded)), (
            "'aborted' trials cannot be 'hit', 'false_alarm', "
            "or 'auto_rewarded'")
        assert not (hit and false_alarm), (
            "both `hit` and `false_alarm` cannot be True, they are mutually "
            "exclusive categories")
        assert not (go and catch), (
            "both `go` and `catch` cannot be True, they are mutually "
            "exclusive "
            "categories")
        assert not (go and auto_rewarded), (
            "both `go` and `auto_rewarded` cannot be True, they are mutually "
            "exclusive categories")

        def _get_response_time(licks: List[float], aborted: bool) -> float:
            """
            Return the time the first lick occurred in a non-"aborted" trial.
            A response time is not returned for on an "aborted trial", since by
            definition, the animal licked before the change stimulus.
            """
            if aborted:
                return float("nan")
            if len(licks):
                return licks[0]
            else:
                return float("nan")

        start_time = event_dict["trial_start", ""]['timestamp']
        stop_time = event_dict["trial_end", ""]['timestamp']

        response_time = _get_response_time(licks, aborted)

        change_frame = self.calculate_change_frame(
                event_dict=event_dict,
                go=go,
                catch=catch,
                auto_rewarded=auto_rewarded)

        result = {
            "start_time": start_time,
            "stop_time": stop_time,
            "trial_length": stop_time - start_time,
            "response_time": response_time,
            "change_frame": change_frame
        }

        result, change_time = self.add_change_time(result)

        if not (go or catch or auto_rewarded):
            response_latency = None
        elif len(licks) > 0:
            response_latency = licks[0] - change_time
        else:
            response_latency = float("inf")

        result["response_latency"] = response_latency

        return result

    def calculate_change_frame(
            self,
            event_dict: dict,
            go: bool,
            catch: bool,
            auto_rewarded: bool):

        """
        Calculate the frame index of a stimulus change
        associated with a specific event.

        Parameters
        ----------
        event_dict: dict
            Dictionary of trial events in the well-known `pkl` file
        go: bool
            True if "go" trial, False otherwise. Mutually exclusive with
            `catch`.
        catch: bool
            True if "catch" trial, False otherwise. Mutually exclusive
            with `go.`
        auto_rewarded: bool
            True if "auto_rewarded" trial, False otherwise.

        Returns
        -------
        change_frame: Union[int, float]
            Index of the change frame; NaN if there is no change

        Notes
        -----
        This is its own method so that child classes of Trial
        can implement different logic as needed.
        """

        if go or auto_rewarded:
            change_frame = event_dict.get(('stimulus_changed', ''))['frame']
        elif catch:
            change_frame = event_dict.get(('sham_change', ''))['frame']
        else:
            change_frame = float("nan")

        return change_frame

    def add_change_time(self, trial_dict: dict) -> Tuple[dict, float]:
        """
        Add change_time to a dict representing a single trial.

        This implementation will just take change_frame and
        select the value of self._stimulus_timestamps corresponding
        to that frame.

        Parameters
        ----------
        trial_dict:
            dict containing all trial parameters except
            change_time

        Returns
        -------
        trial_dict:
            Same as input, except change_time field has been
            added

        change_time: float
            The change time value that was added
            (this is returned separately so that child classes have the
            option of naming the column something different than
            'change_time')

        Note
        ----
        Modified trial_dict in-place, in addition to returning it
        """
        change_frame = trial_dict['change_frame']
        if np.isnan(change_frame):
            change_time = np.nan
        else:
            change_frame = int(change_frame)
            change_time = self._stimulus_timestamps[change_frame]

        trial_dict['change_time'] = change_time
        return trial_dict, change_time

    def _get_trial_image_names(self, stimuli):
        """
        Gets the name of the stimulus presented at the beginning of the
        trial and
        what is it changed to at the end of the trial.
        Parameters
        ----------
        stimuli: The stimuli presentation log for the behavior session

        Returns
        -------
            A dictionary indicating the starting_stimulus and what the
            stimulus is
            changed to.

        """
        grating_oris = {'horizontal', 'vertical'}
        trial_start_frame = self._trial["events"][0][3]
        initial_image_category_name, _, initial_image_name = \
            self._resolve_initial_image(
                stimuli, trial_start_frame)
        if len(self._trial["stimulus_changes"]) == 0:
            change_image_name = initial_image_name
        else:
            ((from_set, from_name),
             (to_set, to_name),
             _, _) = self._trial["stimulus_changes"][0]

            # do this to fix names if the stimuli is a grating
            if from_set in grating_oris:
                from_name = f'gratings_{from_name}'
            if to_set in grating_oris:
                to_name = f'gratings_{to_name}'
            assert from_name == initial_image_name
            change_image_name = to_name

        return {
            "initial_image_name": initial_image_name,
            "change_image_name": change_image_name
        }

    @staticmethod
    def _resolve_initial_image(stimuli, start_frame) -> Tuple[str, str, str]:
        """Attempts to resolve the initial image for a given start_frame for
        a trial

        Parameters
        ----------
        stimuli: Mapping
            foraging2 shape stimuli mapping
        start_frame: int
            start frame of the trial

        Returns
        -------
        initial_image_category_name: str
            stimulus category of initial image
        initial_image_group: str
            group name of the initial image
        initial_image_name: str
            name of the initial image
        """
        max_frame = float("-inf")
        initial_image_group = ''
        initial_image_name = ''
        initial_image_category_name = ''

        for stim_category_name, stim_dict in stimuli.items():
            for set_event in stim_dict["set_log"]:
                set_frame = set_event[3]
                if start_frame >= set_frame >= max_frame:
                    # hack assumes initial_image_group == initial_image_name,
                    # only initial_image_name is present for natual_scenes
                    initial_image_group = initial_image_name = set_event[1]
                    initial_image_category_name = stim_category_name
                    if initial_image_category_name == 'grating':
                        initial_image_name = f'gratings_{initial_image_name}'
                    max_frame = set_frame

        return initial_image_category_name, initial_image_group, \
            initial_image_name

    def _validate_trial_condition_exclusivity(self, tr_data: dict) -> None:
        """ensure that only one of N possible mutually
        exclusive trial conditions is True"""
        trial_conditions = {}
        for key in ['hit',
                    'miss',
                    'false_alarm',
                    'correct_reject',
                    'auto_rewarded',
                    'aborted']:
            trial_conditions[key] = tr_data[key]

        on = []
        for condition, value in trial_conditions.items():
            if value:
                on.append(condition)

        if len(on) != 1:
            all_conditions = list(trial_conditions.keys())
            msg = f"expected exactly 1 trial condition out of " \
                  f"{all_conditions} "
            msg += f"to be True, instead {on} were True (trial {self._index})"
            raise AssertionError(msg)



class Trials(object):

    # @classmethod
    # def trial_class(cls):
    #     """
    #     Return the class to be used to represent a single Trial
    #     """
    #     return Trial

    def __init__(self, trials: pd.DataFrame, response_window_start: float):
        """
        Parameters
        ----------
        trials
        response_window_start
            [seconds] relative to the non-display-lag-compensated presentation
            of the change-image
        """
        trials = trials.rename(columns={"stimulus_change": "is_change"})
        super().__init__(name="trials", value=None, is_value_self=True)
        trials = enforce_df_int_typing(trials, ["change_frame"])

        self._trials = trials
        self._response_window_start = response_window_start

    @classmethod
    def from_stimulus_file(
        cls,
        # pkl_data: dict,
        stimulus_file: BehaviorStimulusFile,
        stimulus_timestamps: np.ndarray,
        licks: pd.DataFrame,
        rewards: pd.DataFrame,
    ) -> "Trials":

        sync_data = read_hdf5_file(self.file_paths['sync_file'], key=None)

        # bsf = pkl_data
        bsf = stimulus_file.data
        stimuli = bsf["items"]["behavior"]["stimuli"]
        trial_log = bsf["items"]["behavior"]["trial_log"]

        trial_bounds = cls._get_trial_bounds(trial_log=trial_log)

        all_trial_data = [None] * len(trial_log)

        for idx, trial in enumerate(trial_log):
            trial_start, trial_end = trial_bounds[idx]
            t = Trial(
                trial=trial,
                start=trial_start,
                end=trial_end,
                pkl_data=bsf,
                index=idx,
                stimulus_timestamps=stimulus_timestamps,
                licks=licks,
                rewards=rewards,
                stimuli=stimuli,
                sync_file=sync_data,
            )

            all_trial_data[idx] = t.data

        trials = pd.DataFrame(all_trial_data).set_index("trial")
        trials.index = trials.index.rename("trials_id")

        # Order/Filter columns
        trials = trials[cls.columns_to_output()]

        return cls(
            trials=trials,
            response_window_start=TaskParameters.from_stimulus_file(
                pkl_data=pkl_data
            ).response_window_sec[0],
        )

    @property
    def data(self) -> pd.DataFrame:
        return self._trials

    @property
    def trial_count(self) -> int:
        """Number of trials
        (including all 'go', 'catch', and 'aborted' trials)"""
        return self._trials.shape[0]

    @property
    def go_trial_count(self) -> int:
        """Number of 'go' trials"""
        return self._trials["go"].sum()

    @property
    def catch_trial_count(self) -> int:
        """Number of 'catch' trials"""
        return self._trials["catch"].sum()

    @property
    def hit_trial_count(self) -> int:
        """Number of trials with a hit behavior response"""
        return self._trials["hit"].sum()

    @property
    def miss_trial_count(self) -> int:
        """Number of trials with a hit behavior response"""
        return self._trials["miss"].sum()

    @property
    def false_alarm_trial_count(self) -> int:
        """Number of trials where the mouse had a false alarm
        behavior response"""
        return self._trials["false_alarm"].sum()

    @property
    def correct_reject_trial_count(self) -> int:
        """Number of trials with a correct reject behavior
        response"""
        return self._trials["correct_reject"].sum()

    @classmethod
    def columns_to_output(cls) -> List[str]:
        """
        Return the list of columns to be output in this table
        """
        return [
            "initial_image_name",
            "change_image_name",
            "stimulus_change",
            "change_time",
            "go",
            "catch",
            "lick_times",
            "response_time",
            "response_latency",
            "reward_time",
            "reward_volume",
            "hit",
            "false_alarm",
            "miss",
            "correct_reject",
            "aborted",
            "auto_rewarded",
            "change_frame",
            "start_time",
            "stop_time",
            "trial_length",
        ]

    
    @staticmethod
    def _get_trial_bounds(trial_log: List):
        """
        Adjust trial boundaries from a trial_log so that there is no dead time
        between trials.

        Parameters
        ----------
        trial_log: list
            The trial_log read in from the well known behavior stimulus
            pickle file

        Returns
        -------
        list
            Each element in the list is a tuple of the form
            (start_frame, end_frame) so that the ith element
            of the list gives the start and end frames of
            the ith trial. The endframe of the last trial will
            be -1, indicating that it should map to the last
            timestamp in the session
        """
        start_frames = []

        for trial in trial_log:
            start_f = None
            for event in trial["events"]:
                if event[0] == "trial_start":
                    start_f = event[-1]
                    break
            if start_f is None:
                msg = "Could not find a 'trial_start' event "
                msg += "for all trials in the trial log\n"
                msg += f"{trial}"
                raise ValueError(msg)

            if len(start_frames) > 0 and start_f < start_frames[-1]:
                msg = "'trial_start' frames in trial log "
                msg += "are not in ascending order"
                msg += f"\ntrial_log: {trial_log}"
                raise ValueError(msg)

            start_frames.append(start_f)

        end_frames = [idx for idx in start_frames[1:] + [-1]]
        return list([(s, e) for s, e in zip(start_frames, end_frames)])

    @property
    def index(self) -> pd.Index:
        return self.data.index

    @property
    def change_time(self) -> pd.Series:
        if "change_time" in self.data:
            return self.data["change_time"]
        elif "change_time_no_display_delay" in self.data:
            return self.data["change_time_no_display_delay"]

    @property
    def lick_times(self) -> pd.Series:
        return self.data["lick_times"]

    @property
    def start_time(self) -> pd.Series:
        return self.data["start_time"]

    @property
    def aborted(self) -> pd.Series:
        return self.data["aborted"]

    @property
    def hit(self) -> pd.Series:
        return self.data["hit"]

    @property
    def miss(self) -> pd.Series:
        return self.data["miss"]

    @property
    def false_alarm(self) -> pd.Series:
        return self.data["false_alarm"]

    @property
    def correct_reject(self) -> pd.Series:
        return self.data["correct_reject"]

    @property
    def rolling_performance(self) -> pd.DataFrame:
        """Return a DataFrame containing trial by trial behavior response
        performance metrics.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing:
                trials_id [index]:
                    Index of the trial. All trials, including aborted trials,
                    are assigned an index starting at 0 for the first trial.
                reward_rate:
                    Rewards earned in the previous 25 trials, normalized by
                    the elapsed time of the same 25 trials. Units are
                    rewards/minute.
                hit_rate_raw:
                    Fraction of go trials where the mouse licked in the
                    response window, calculated over the previous 100
                    non-aborted trials. Without trial count correction applied.
                hit_rate:
                    Fraction of go trials where the mouse licked in the
                    response window, calculated over the previous 100
                    non-aborted trials. With trial count correction applied.
                false_alarm_rate_raw:
                    Fraction of catch trials where the mouse licked in the
                    response window, calculated over the previous 100
                    non-aborted trials. Without trial count correction applied.
                false_alarm_rate:
                    Fraction of catch trials where the mouse licked in
                    the response window, calculated over the previous 100
                    non-aborted trials. Without trial count correction applied.
                rolling_dprime:
                    d prime calculated using the rolling hit_rate and
                    rolling false_alarm _rate.

        """
        reward_rate = self.calculate_reward_rate()

        # Indices to build trial metrics dataframe:
        trials_index = self.data.index
        not_aborted_index = self.data[np.logical_not(self.aborted)].index

        # Initialize dataframe:
        performance_metrics_df = pd.DataFrame(index=trials_index)

        # Reward rate:
        performance_metrics_df["reward_rate"] = pd.Series(
            reward_rate, index=self.data.index
        )

        # Hit rate raw:
        hit_rate_raw = get_hit_rate(
            hit=self.hit, miss=self.miss, aborted=self.aborted
        )
        performance_metrics_df["hit_rate_raw"] = pd.Series(
            hit_rate_raw, index=not_aborted_index
        )

        # Hit rate with trial count correction:
        hit_rate = get_trial_count_corrected_hit_rate(
            hit=self.hit, miss=self.miss, aborted=self.aborted
        )
        performance_metrics_df["hit_rate"] = pd.Series(
            hit_rate, index=not_aborted_index
        )

        # False-alarm rate raw:
        false_alarm_rate_raw = get_false_alarm_rate(
            false_alarm=self.false_alarm,
            correct_reject=self.correct_reject,
            aborted=self.aborted,
        )
        performance_metrics_df["false_alarm_rate_raw"] = pd.Series(
            false_alarm_rate_raw, index=not_aborted_index
        )

        # False-alarm rate with trial count correction:
        false_alarm_rate = get_trial_count_corrected_false_alarm_rate(
            false_alarm=self.false_alarm,
            correct_reject=self.correct_reject,
            aborted=self.aborted,
        )
        performance_metrics_df["false_alarm_rate"] = pd.Series(
            false_alarm_rate, index=not_aborted_index
        )

        # Rolling-dprime:
        is_passive_session = (self.data["reward_volume"] == 0).all() and (
            self.data["lick_times"].apply(lambda x: len(x)) == 0
        ).all()
        if is_passive_session:
            # It does not make sense to calculate d' for a passive session
            # So just set it to zeros
            rolling_dprime = np.zeros(len(hit_rate))
        else:
            rolling_dprime = get_rolling_dprime(hit_rate, false_alarm_rate)
        performance_metrics_df["rolling_dprime"] = pd.Series(
            rolling_dprime, index=not_aborted_index
        )

        return performance_metrics_df

    def _calculate_response_latency_list(self) -> List:
        """per trial, determines a response latency

        Returns
        -------
        response_latency_list: List
            len() = trials.shape[0]
            value is 'inf' if there are no valid licks in the trial

        Note
        -----
        response_window_start is listed as
        "relative to the non-display-lag-compensated..." because it
        comes directly from the stimulus file, which knows nothing
        about the display lag. However, response_window_start is
        only ever compared to the difference between
        trial.lick_times and trial.change_time, both of which are
        corrected for monitor delay, so it does not matter
        (the two instance of monitor delay cancel out in the
        difference).
        """
        df = pd.DataFrame(
            {"lick_times": self.lick_times, "change_time": self.change_time}
        )
        df["valid_response_licks"] = df.apply(
            lambda trial: [
                lt
                for lt in trial["lick_times"]
                if lt - trial["change_time"] > self._response_window_start
            ],
            axis=1,
        )
        response_latency = df.apply(
            lambda trial: trial["valid_response_licks"][0]
            - trial["change_time"]
            if len(trial["valid_response_licks"]) > 0
            else float("inf"),
            axis=1,
        )
        return response_latency.tolist()

    def calculate_reward_rate(
        self, window=0.75, trial_window=25, initial_trials=10
    ):
        response_latency = self._calculate_response_latency_list()
        starttime = self.start_time.values
        assert len(response_latency) == len(starttime)

        df = pd.DataFrame(
            {"response_latency": response_latency, "starttime": starttime}
        )

        # adds a column called reward_rate to the input dataframe
        # the reward_rate column contains a rolling average of rewards/min
        # window sets the window in which a response is considered correct,
        # so a window of 1.0 means licks before 1.0 second are considered
        # correct

        # Reorganized into this unit-testable form by Nick Cain April 25 2019

        reward_rate = np.zeros(len(df))
        # make the initial reward rate infinite,
        # so that you include the first trials automatically.
        reward_rate[:initial_trials] = np.inf

        for trial_number in range(initial_trials, len(df)):
            min_index = np.max((0, trial_number - trial_window))
            max_index = np.min((trial_number + trial_window, len(df)))
            df_roll = df.iloc[min_index:max_index]

            # get a rolling number of correct trials
            correct = len(df_roll[df_roll.response_latency < window])

            # get the time elapsed over the trials
            time_elapsed = (
                df_roll.starttime.iloc[-1] - df_roll.starttime.iloc[0]
            )

            # calculate the reward rate, rewards/min
            reward_rate_on_this_lap = correct / time_elapsed * 60

            reward_rate[trial_number] = reward_rate_on_this_lap

        reward_rate[np.isinf(reward_rate)] = float("nan")
        return reward_rate

    def _get_engaged_trials(
        self, engaged_trial_reward_rate_threshold: float = 2.0
    ) -> pd.Series:
        """
        Gets `Series` where each trial that is considered "engaged" is set to
        `True`

        Parameters
        ----------
        engaged_trial_reward_rate_threshold:
            The number of rewards per minute that needs to be attained
            before a subject is considered 'engaged', by default 2.0

        Returns
        -------
        `pd.Series`
        """
        rolling_performance = self.rolling_performance
        engaged_trial_mask = (
            rolling_performance["reward_rate"]
            > engaged_trial_reward_rate_threshold
        )
        return engaged_trial_mask

    def get_engaged_trial_count(
        self, engaged_trial_reward_rate_threshold: float = 2.0
    ) -> int:
        """Gets count of trials considered "engaged"

        Parameters
        ----------
        engaged_trial_reward_rate_threshold:
            The number of rewards per minute that needs to be attained
            before a subject is considered 'engaged', by default 2.0

        Returns
        -------
        count of trials considered "engaged"
        """
        engaged_trials = self._get_engaged_trials(
            engaged_trial_reward_rate_threshold=(
                engaged_trial_reward_rate_threshold
            )
        )
        return engaged_trials.sum()




    @staticmethod
    def _calculate_stimulus_duration(
            pkl_data: dict):
        data = pkl_data

        behavior = data["items"]["behavior"]
        stimuli = behavior['stimuli']

        def _parse_stimulus_key():
            if 'images' in stimuli:
                stim_key = 'images'
            elif 'grating' in stimuli:
                stim_key = 'grating'
            else:
                msg = "Cannot get stimulus_duration_sec\n"
                msg += "'images' and/or 'grating' not a valid "
                msg += "key in pickle file under "
                msg += "['items']['behavior']['stimuli']\n"
                msg += f"keys: {list(stimuli.keys())}"
                raise RuntimeError(msg)

            return stim_key
        stim_key = _parse_stimulus_key()
        stim_duration = stimuli[stim_key]['flash_interval_sec']

        # from discussion in
        # https://github.com/AllenInstitute/AllenSDK/issues/1572
        #
        # 'flash_interval' contains (stimulus_duration, gray_screen_duration)
        # (as @matchings said above). That second value is redundant with
        # 'blank_duration_range'. I'm not sure what would happen if they were
        # set to be conflicting values in the params. But it looks like
        # they're always consistent. It should always be (0.25, 0.5),
        # except for TRAINING_0 and TRAINING_1, which have statically
        # displayed stimuli (no flashes).

        if stim_duration is None:
            stim_duration = np.NaN
        else:
            stim_duration = stim_duration[0]
        return stim_duration

    @staticmethod
    def _calculuate_n_stimulus_frames(
            pkl_data: dict) -> int:
        data = pkl_data
        behavior = data["items"]["behavior"]

        n_stimulus_frames = 0
        for stim_type, stim_table in behavior["stimuli"].items():
            n_stimulus_frames += sum(stim_table.get("draw_log", []))
        return n_stimulus_frames


class BehaviorStimulusType():
    IMAGES = 'images'
    GRATING = 'grating'


class StimulusDistribution():
    EXPONENTIAL = 'exponential'
    GEOMETRIC = 'geometric'


class TaskType():
    CHANGE_DETECTION = 'change detection'

class TaskParameters():
    def __init__(self,
                 blank_duration_sec: List[float],
                 stimulus_duration_sec: float,
                 omitted_flash_fraction: float,
                 response_window_sec: List[float],
                 reward_volume: float,
                 auto_reward_volume: float,
                 session_type: str,
                 stimulus: str,
                 stimulus_distribution: StimulusDistribution,
                 task_type: TaskType,
                 n_stimulus_frames: int,
                 stimulus_name: Optional[str] = None):
        super().__init__(name='task_parameters', value=None,
                         is_value_self=True)
        self._blank_duration_sec = blank_duration_sec
        self._stimulus_duration_sec = stimulus_duration_sec
        self._omitted_flash_fraction = omitted_flash_fraction
        self._response_window_sec = response_window_sec
        self._reward_volume = reward_volume
        self._auto_reward_volume = auto_reward_volume
        self._session_type = session_type
        self._stimulus = BehaviorStimulusType(stimulus)
        self._stimulus_distribution = StimulusDistribution(
            stimulus_distribution)
        self._task = TaskType(task_type)
        self._n_stimulus_frames = n_stimulus_frames
        self._stimulus_name = stimulus_name
        self._image_set = parse_stimulus_set(session_type)

    @property
    def blank_duration_sec(self) -> List[float]:
        return self._blank_duration_sec

    @property
    def stimulus_duration_sec(self) -> float:
        return self._stimulus_duration_sec

    @property
    def omitted_flash_fraction(self) -> float:
        return self._omitted_flash_fraction

    @property
    def response_window_sec(self) -> List[float]:
        return self._response_window_sec

    @property
    def reward_volume(self) -> float:
        return self._reward_volume

    @property
    def auto_reward_volume(self) -> float:
        return self._auto_reward_volume

    @property
    def session_type(self) -> str:
        return self._session_type

    @property
    def stimulus(self) -> str:
        return self._stimulus

    @property
    def stimulus_distribution(self) -> float:
        return self._stimulus_distribution

    @property
    def task(self) -> TaskType:
        return self._task

    @property
    def n_stimulus_frames(self) -> int:
        return self._n_stimulus_frames

    @property
    def stimulus_name(self) -> Optional[str]:
        return self._stimulus_name

    @property
    def image_set(self) -> str:
        return self._image_set


    @classmethod
    def from_stimulus_file(
            cls,
            pkl_data: dict) -> "TaskParameters":
        data = pkl_data

        behavior = data["items"]["behavior"]
        config = behavior["config"]
        doc = config["DoC"]

        blank_duration_sec = [float(x) for x in doc['blank_duration_range']]
        stim_duration = cls._calculate_stimulus_duration(
            pkl_data=pkl_data)
        omitted_flash_fraction = \
            behavior['params'].get('flash_omit_probability', float('nan'))
        response_window_sec = [float(x) for x in doc["response_window"]]
        reward_volume = config["reward"]["reward_volume"]
        auto_reward_volume = doc['auto_reward_volume']
        session_type = behavior["params"]["stage"]
        stimulus = next(iter(behavior["stimuli"]))
        # stimulus_name = stimulus_file.stimulus_name
        stimulus_distribution = doc["change_time_dist"]
        task = cls._parse_task(pkl_data=pkl_data)
        n_stimulus_frames = cls._calculuate_n_stimulus_frames(
            pkl_data=pkl_data)
        return TaskParameters(
            blank_duration_sec=blank_duration_sec,
            stimulus_duration_sec=stim_duration,
            omitted_flash_fraction=omitted_flash_fraction,
            response_window_sec=response_window_sec,
            reward_volume=reward_volume,
            auto_reward_volume=auto_reward_volume,
            session_type=session_type,
            stimulus=stimulus,
            stimulus_distribution=stimulus_distribution,
            task_type=task,
            n_stimulus_frames=n_stimulus_frames,
            stimulus_name=stimulus_name
        )

    @staticmethod
    def _calculate_stimulus_duration(
            pkl_data: dict) -> float:
        data = pkl_data

        behavior = data["items"]["behavior"]
        stimuli = behavior['stimuli']

        def _parse_stimulus_key():
            if 'images' in stimuli:
                stim_key = 'images'
            elif 'grating' in stimuli:
                stim_key = 'grating'
            else:
                msg = "Cannot get stimulus_duration_sec\n"
                msg += "'images' and/or 'grating' not a valid "
                msg += "key in pickle file under "
                msg += "['items']['behavior']['stimuli']\n"
                msg += f"keys: {list(stimuli.keys())}"
                raise RuntimeError(msg)

            return stim_key
        stim_key = _parse_stimulus_key()
        stim_duration = stimuli[stim_key]['flash_interval_sec']

        # from discussion in
        # https://github.com/AllenInstitute/AllenSDK/issues/1572
        #
        # 'flash_interval' contains (stimulus_duration, gray_screen_duration)
        # (as @matchings said above). That second value is redundant with
        # 'blank_duration_range'. I'm not sure what would happen if they were
        # set to be conflicting values in the params. But it looks like
        # they're always consistent. It should always be (0.25, 0.5),
        # except for TRAINING_0 and TRAINING_1, which have statically
        # displayed stimuli (no flashes).

        if stim_duration is None:
            stim_duration = np.NaN
        else:
            stim_duration = stim_duration[0]
        return stim_duration

    @staticmethod
    def _parse_task(
            pkl_data: dict) -> TaskType:
        data = pkl_data
        config = data["items"]["behavior"]["config"]

        task_id = config['behavior']['task_id']
        if 'DoC' in task_id:
            task = TaskType.CHANGE_DETECTION
        else:
            msg = "metadata.get_task_parameters does not "
            msg += f"know how to parse 'task_id' = {task_id}"
            raise RuntimeError(msg)
        return task

    @staticmethod
    def _calculuate_n_stimulus_frames(
            pkl_data: dict) -> int:
        data = pkl_data
        behavior = data["items"]["behavior"]

        n_stimulus_frames = 0
        for stim_type, stim_table in behavior["stimuli"].items():
            n_stimulus_frames += sum(stim_table.get("draw_log", []))
        return n_stimulus_frames