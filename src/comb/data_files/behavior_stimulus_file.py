import copy
import datetime
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any

import pandas as pd


class BehaviorStimulusFile(object):
    def __init__(self, data: Any):
        self._data = data
        self._filepath = None
        self._behavior_key = None

    @property
    def data(self) -> Any:  # pragma: no cover
        return self._data

    @property
    def filepath(self) -> str:  # pragma: no cover
        return self._filepath

    @property
    def behavior_key(self) -> str:
        return self._behavior_key

    @classmethod
    def from_file(cls, file_path: str) -> "BehaviorStimulusFile":
        data = pd.read_pickle(file_path)
        # set filepath attribute to class instance
        instance = cls(data)
        instance._filepath = file_path  # DataObject (allenSDK) abstract class has a _filepath attribute, here we set it.
        instance._behavior_key = instance._get_behavior_key()

        return instance

    def _get_behavior_key(self):
        """ data: loaded pkl_file """
        # TODO: check logic of passive v active
        # behavior_keys = ['behavior', 'foraging']

        if 'behavior' in self.data['items']:
            behavior_key = 'behavior'
        elif 'foraging' in self.data['items']:
            behavior_key = 'foraging'

        return behavior_key

    # # NOTE: unclear why this validation is needed @mattjdavis
    # def _validate_frame_data(self):
    #     """
    #     Make sure that self.data['intervalsms'] does not exist and that
    #     self.data['items']['behavior']['intervalsms'] does exist.
    #     """
    #     msg = ""
    #     if "intervalsms" in self.data:
    #         msg += "self.data['intervalsms'] present; did not expect that\n"
    #     if "items" not in self.data:
    #         msg += "self.data['items'] not present\n"
    #     else:
    #         if self.behavior_key not in self.data["items"]:
    #             msg += "self.data['items'][BEHAVIOR_KEY] not present\n"
    #         else:
    #             if "intervalsms" not in self.data["items"][self.behavior_key]:
    #                 msg += (
    #                     "self.data['items'][BEHAVIOR_KEY]['intervalsms'] "
    #                     "not present\n"
    #                 )

    #     if len(msg) > 0:
    #         full_msg = f"When getting num_frames from {type(self)}\n"
    #         full_msg += msg
    #         full_msg += f"\nfilepath: {self.filepath}"
    #         raise RuntimeError(full_msg)

    #     return None

    @property
    def behavior_session_uuid(self) -> Optional[uuid.UUID]:
        """Return the behavior session UUID either from the uuid field
        or foraging_id field.
        """
        bs_uuid = self.data.get("session_uuid")
        if bs_uuid is None:
            try:
                bs_uuid = self._retrieve_from_params("foraging_id")["value"]
            except (KeyError, RuntimeError):
                bs_uuid = None
        if bs_uuid:
            try:
                bs_uuid = uuid.UUID(bs_uuid)
            except ValueError:
                bs_uuid = None
        return bs_uuid

    @property
    def date_of_acquisition(self) -> datetime.datetime:
        """
        Return the date_of_acquisition as a datetime.datetime.

        This will be read from self.data['start_time']
        """
        assert isinstance(self.data, dict)
        if "start_time" not in self.data:
            raise KeyError(
                "No 'start_time' listed in pickle file " f"{self.filepath}"
            )

        return copy.deepcopy(self.data["start_time"])

    @property
    def mouse_id(self) -> str:
        """Retrieve the mouse_id value from the stimulus pickle file.

        This can be read either from:

        data['items'][behavior_key]['params']['stage']
        or
        data['items'][behavior_key]['cl_params']['stage']

        if both are present and they disagree, raise an exception.
        """
        return self._retrieve_from_params("mouse_id")

    @property
    def num_frames(self) -> int:
        """
        Return the number of frames associated with this StimulusFile
        """
        # NOTE: changed key here @mattjdavis
        # self._validate_frame_data()
        # return len(self.data["items"]["behavior"]["intervalsms"]) + 1

        return len(self.data["intervalsms"]) + 1

    @property
    def session_type(self) -> str:
        """
        Return the session type as read from the pickle file. This can
        be read either from

        data['items'][behavior_key]['params']['stage']
        or
        data['items'][behavior_key]['cl_params']['stage']

        if both are present and they disagree, raise an exception
        """
        return self._retrieve_from_params("stage")

    @property
    def stimulus_name(self) -> str:
        """
        Get the image stimulus name by parsing the file path of the image set.

        If no image set, check for gratings and return "behavior" if not found.

        Parameters
        ----------
        stimulus_file : BehaviorStimulusFile
            Stimulus pickle file to parse.

        Returns
        -------
        stimulus_name : str
            Name of the image stimulus from the image file path set shown to
            the mouse.
        """
        # MJD hack for omfish
        if isinstance(self.stimuli, list):
            stimulus_name = "passive behavior (found list)"
            return stimulus_name
        try:
            stimulus_name = Path(
                self.stimuli["images"]["image_set"]
            ).stem.split(".")[0]
        except KeyError:
            # if we can't find the images key in the stimuli, check for the
            # name ``grating`` as the stimulus. If not add generic
            # ``behavior``.
            if "grating" in self.stimuli.keys():
                stimulus_name = "grating"
            else:
                stimulus_name = "behavior"
        return stimulus_name

    def _retrieve_from_params(self, key_name: str):
        """Retrieve data from either data['items'][behavior_key]['params'] or
        data['items'][behavior_key]['cl_params'].

        Test for conflicts or missing data and raise if issues found.

        Parameters
        ----------
        key_name : str
            Name of data to attempt to retrieve from the behavior stimulus
            file data.

        Returns
        -------
        value : various
            Value from the stimulus file.
        """
        behavior_key = self.behavior_key
        param_value = None
        if "params" in self.data["items"][behavior_key]:
            if key_name in self.data["items"][behavior_key]["params"]:
                param_value = self.data["items"][behavior_key]["params"][
                    key_name
                ]

        cl_value = None
        if "cl_params" in self.data["items"][behavior_key]:
            if key_name in self.data["items"][behavior_key]["cl_params"]:
                cl_value = self.data["items"][behavior_key]["cl_params"][
                    key_name
                ]

        if cl_value is None and param_value is None:
            raise RuntimeError(
                f"Could not find {key_name} in pickle file " f"{self.filepath}"
            )

        if param_value is None:
            return cl_value

        if cl_value is None:
            return param_value

        if cl_value != param_value:
            raise RuntimeError(
                f"Conflicting {key_name} values in pickle file "
                f"{self.filepath}\n"
                f"cl_params: {cl_value}\n"
                f"params: {param_value}\n"
            )

        return param_value

    @property
    def session_duration(self) -> float:
        """
        Gets session duration in seconds

        Returns
        -------
        session duration in seconds
        """
        start_time = self.data["start_time"]
        stop_time = self.data["stop_time"]

        if not isinstance(start_time, datetime.datetime):
            start_time = datetime.datetime.fromtimestamp(start_time)
        if not isinstance(stop_time, datetime.datetime):
            stop_time = datetime.datetime.fromtimestamp(stop_time)

        delta = stop_time - start_time

        return delta.total_seconds()

    @property
    def stimuli(self) -> Dict[str, Tuple[str, Union[str, int], int, int]]:
        """Stimuli shown during session

        Returns
        -------
        stimuli:
            (stimulus type ('Image' or 'Grating'),
             stimulus descriptor (image_name or orientation of grating in
                degrees),
             nonsynced time of display,
             display frame (frame that stimuli was displayed))

        """
        # TODO implement return value as class (i.e. Image, Grating)
        if self.behavior_key == "behavior":
            return self.data["items"][self.behavior_key]["stimuli"]
        elif self.behavior_key == "foraging":
            return self.data["stimuli"]

    def validate(self) -> "BehaviorStimulusFile":
        if "items" not in self.data or "behavior" not in self.data["items"]:
            raise MalformedStimulusFileError(
                f'Expected to find key "behavior" in "items" dict. '
                f'Found {self.data["items"].keys()}'
            )
        return self


class MalformedStimulusFileError(RuntimeError):
    """Malformed stimulus file"""

    pass