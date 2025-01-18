import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from pynwb import NWBFile, TimeSeries



from comb.core import DataObject 
from comb.processing.timestamps.stimulus_timestamps import StimulusTimestamps
from comb.data_files.eye_tracking_file import EyeTrackingFile

    

# metadata file ignored in COMB!  
# from allensdk.brain_observatory.behavior.\
#     data_files.eye_tracking_metadata_file import EyeTrackingMetadataFile

# from allensdk.brain_observatory.behavior.eye_tracking_processing import \
#     process_eye_tracking_data
    


class EyeTrackingTable(DataObject):
    """corneal, eye, and pupil ellipse fit data"""
    _logger = logging.getLogger(__name__)

    def __init__(self, eye_tracking: pd.DataFrame):
        super().__init__(name='eye_tracking', value=eye_tracking)

    @classmethod
    def _get_empty_df(cls) -> pd.DataFrame:
        """
        Return an empty dataframe with the correct column and index
        names, but no data
        """
        raise NotImplementedError("This method is not implemented, Code ocean cols are diff then AllenSDK cols")
    
        empty_data = dict()
        
        # for colname in ['timestamps', 'cr_area', 'eye_area',
        #                 'pupil_area', 'likely_blink', 'pupil_area_raw',
        #                 'cr_area_raw', 'eye_area_raw', 'cr_center_x',
        #                 'cr_center_y', 'cr_width', 'cr_height', 'cr_phi',
        #                 'eye_center_x', 'eye_center_y', 'eye_width',
        #                 'eye_height', 'eye_phi', 'pupil_center_x',
        #                 'pupil_center_y', 'pupil_width', 'pupil_height',
        #                 'pupil_phi']:
        #     empty_data[colname] = []

        eye_tracking_data = pd.DataFrame(empty_data,
                                         index=pd.Index([], name='frame'))
        return eye_tracking_data

    
    @classmethod
    def from_data_file(
            cls,
            #data_file: EyeTrackingFile,
            data_file: pd.DataFrame,
            stimulus_timestamps: StimulusTimestamps,
            #metadata_file: Optional[EyeTrackingMetadataFile] = None,
            #z_threshold: float = 3.0,
            #dilation_frames: int = 2,
            empty_on_fail: bool = False):
        """
        Parameters
        ----------
        data_file
        stimulus_timestamps: StimulusTimestamps
            The timestamps associated with this eye tracking table
        z_threshold : float, optional
            See EyeTracking.from_lims
        dilation_frames : int, optional
             See EyeTracking.from_lims
        empty_on_fail: bool
            If True, this method will return an empty dataframe
            if an EyeTrackingError is raised (usually because
            timestamps and eye tracking video frames do not
            align). If false, the error will get raised.
        metadata_file: EyeTrackingMetadataFile. Used for detecting if video is
            MVR. Either this or video must be given.
        video: EyeTrackingVideo. Used for detecting if video is MVR.
            Either this or metadata_file must be given.
        """
        # cls._logger.info(f"Getting eye_tracking_data with "
        #                  f"'z_threshold={z_threshold}', "
        #                  f"'dilation_frames={dilation_frames}'")


        # is_metadata_frame_present = (
        #     _is_metadata_frame_present(
        #         metadata_file=metadata_file,
        #         video=video
        #     ) if metadata_file is not None or video is not None else False)
        
        # assume always true for learning mfish (MJD 1/17/2025)
        is_metadata_frame_present = True

        try:
            frames, stimulus_timestamps = cls._validate_frame_time_alignment(
                #frames=data_file.data.index.values, times=stimulus_timestamps, # HACK, no data file
                frames=data_file.index.values, times=stimulus_timestamps,
                is_metadata_frame_present=is_metadata_frame_present
            )
            print(frames,stimulus_timestamps)
            # eye_data = data_file.data.loc[frames] # HACK no data file
            eye_data = data_file.loc[frames]

            if is_metadata_frame_present:
                # Reset index to start at 0 if metadata frame was dropped
                eye_data.index -= 1

            # IN CodeOCean, the dlc capsule already processes the tables.
     
            eye_tracking_data = process_eye_tracking_data(
                                     eye_data,
                                     stimulus_timestamps.value)
        except Exception as err:
            if empty_on_fail:
                msg = f"{str(err)}\n"
                msg += "returning empty eye_tracking DataFrame"
                warnings.warn(msg)
                eye_tracking_data = cls._get_empty_df()
            else:
                raise

        #return EyeTrackingTable(eye_tracking=eye_tracking_data)
        return eye_tracking_data

    @classmethod
    def _validate_frame_time_alignment(
            cls,
            frames: np.ndarray,
            times: StimulusTimestamps,
            is_metadata_frame_present: bool = False
    ) -> Tuple[np.ndarray, StimulusTimestamps]:
        """
        Checks whether frames or timestamps need to be modified in order to be
            aligned with each other. If so, does the alignment.

        Algorithm:
        1. Remove metadata frame, if present
        2. If # frames > # timestamps: raise error
           else if # timestamps > # frames: truncate frames

        Parameters
        ----------
        frames: eye tracking frames
        times: eye tracking timestamps
        is_metadata_frame_present: Whether frames contains a metadata frame as
            the first frame

        Returns
        -------
        Tuple of frames, timestamps, where frames and timestamps have been
            corrected to be aligned with each other
        """
        if is_metadata_frame_present:
            # Remove the metadata frame
            cls._logger.info(
                f'Number of eye tracking timestamps: {len(times.value)}. '
                f'Number of eye tracking frames: {len(frames)}. '
                f'Removing metadata frame')
            frames = frames[1:]

        if len(times.value) > len(frames): # MJD: times -> times.value
            # It's possible for there to be more timestamps than frames in a
            # case of non-transferred frames/aborted frames
            # See discussion in https://github.com/AllenInstitute/AllenSDK/issues/2376 # noqa
            # Truncate timestamps to match the number of frames
            cls._logger.info(
                f'Number of eye tracking timestamps: {len(times.value)}. '
                f'Number of eye tracking frames: {len(frames)}. '
                f'Truncating timestamps')
            times = times.update_timestamps(
                timestamps=times.value[:len(frames)])
        elif len(frames) > len(times.value):
            print(
                f'Number of eye tracking timestamps: {len(times.value)}. '
                f'Number of eye tracking frames: {len(frames)}. '
                f'We expect these to be equal')
        return frames, times


# COMB: not sure when called

# def get_lost_frames(
#         eye_tracking_metadata: EyeTrackingMetadataFile) -> np.ndarray: # REPLACE: 
#     """
#     Get lost frames from the video metadata json
#     Must subtract one since the json starts indexing at 1

#     The lost frames are recorded like
#     ['13-14,67395-67398']
#     which would mean frames 13, 15, 67395, 67396, 67397, 67398
#     were lost.

#     This method needs to parse these strings into lists of integers.

#     Parameters
#     ----------
#     eye_tracking_metadata: EyeTrackingMetadataFile

#     Returns
#     -------
#         indices of lost frames

#     Notes
#     -----
#     This algorithm was copied almost directly from an implementation at
#     https://github.com/corbennett/NP_pipeline_QC/blob/6a66f195c4cd6b300776f089773577db542fe7eb/probeSync_qc.py
#     """

#     camera_metadata = eye_tracking_metadata.data

#     lost_count = camera_metadata['RecordingReport']['FramesLostCount']
#     if lost_count == 0:
#         return []

#     lost_string = camera_metadata['RecordingReport']['LostFrames'][0]
#     lost_spans = lost_string.split(',')

#     lost_frames = []
#     for span in lost_spans:
#         start_end = span.split('-')
#         if len(start_end) == 1:
#             lost_frames.append(int(start_end[0]))
#         else:
#             lost_frames.extend(np.arange(int(start_end[0]),
#                                          int(start_end[1])+1))

#     return np.array(lost_frames)-1


# COMB assume always present, could fix later

# def _is_metadata_frame_present(
#         metadata_file: Optional[EyeTrackingMetadataFile] = None,
#         video: Optional[EyeTrackingVideo] = None
# ) -> bool:
#     """Return whether a metadata frame was placed at the front of the eye
#     tracking movie. Tries to determine this by using the fact that the MVR
#     (multi-video-recorder) software always places a metadata frame at the
#     front. Detect MVR by the filetype. MVR outputs mp4 while predecessors
#     output a different format.

#     First checks the metadata file if given for the video filepath
#     Then checks video file if metadata file not given

#     Raises
#     ------
#     ValueError if neither metadata_file or video is given
#     """
#     if metadata_file is not None:
#         video_file_name = \
#             metadata_file.data['RecordingReport']['VideoOutputFileName']\
#             .lower()
#     elif video is not None:
#         video_file_name = video.filepath.lower()
#     else:
#         raise ValueError('Either metadata_file or video must be given')
#     video_file_name = Path(video_file_name)

#     return video_file_name.suffix == '.mp4' or \
#         'mvr' in video_file_name.name



def process_eye_tracking_data(eye_data: pd.DataFrame,
                              frame_times: pd.Series,
                              z_threshold: float = 3.0,
                              dilation_frames: int = 2) -> pd.DataFrame:
    """Processes and refines raw eye tracking data by adding additional
    computed feature columns.

    Parameters
    ----------
    eye_data : pd.DataFrame
        A 'raw' eye tracking dataframe produced by load_eye_tracking_hdf()
    frame_times : pd.Series
        A series of frame times acquired from a behavior + ophy session
        'sync file'.
    z_threshold : float
        z-score values higher than the z_threshold will be considered outliers,
        by default 3.0.
    dilation_frames : int, optional
        Determines the number of additional adjacent frames to mark as
        'likely_blink', by default 2.

    Returns
    -------
    pd.DataFrame
        A refined eye tracking dataframe that contains additional information
        about frame times, eye areas, pupil areas, and frames with likely
        blinks/outliers.

    Raises
    ------
    EyeTrackingError
        If the number of sync file frame times does not match the number of
        eye tracking frames.
    """

    n_sync = len(frame_times)
    n_eye_frames = len(eye_data.index)

    # If n_sync exceeds n_eye_frames by <= 15,
    # just trim the excess sync pulses from the end
    # of the timestamps array.
    #
    # This solution was discussed in
    # https://github.com/AllenInstitute/AllenSDK/issues/1545

    if n_eye_frames < n_sync <= n_eye_frames + 15:
        frame_times = frame_times[:n_eye_frames]
        n_sync = len(frame_times)

    if n_sync != n_eye_frames:
        raise(f"Error! The number of sync file frame times "
                               f"({len(frame_times)}) does not match the "
                               f"number of eye tracking frames "
                               f"({len(eye_data.index)})!")

    # cr_areas = (eye_data[["cr_width", "cr_height"]]
    #             .apply(compute_elliptical_area, axis=1))
    # eye_areas = (eye_data[["eye_width", "eye_height"]]
    #              .apply(compute_elliptical_area, axis=1))
    # pupil_areas = (eye_data[["pupil_width", "pupil_height"]]
    #                .apply(compute_circular_area, axis=1))

    # # only use eye and pupil areas for outlier detection
    # area_df = pd.concat([eye_areas, pupil_areas], axis=1)
    # outliers = determine_outliers(area_df, z_threshold=z_threshold)

    # likely_blinks = determine_likely_blinks(eye_areas,
    #                                         pupil_areas,
    #                                         outliers,
    #                                         dilation_frames=dilation_frames)

    # # remove outliers/likely blinks `pupil_area`, `cr_area`, `eye_area`
    # pupil_areas_raw = pupil_areas.copy()
    # cr_areas_raw = cr_areas.copy()
    # eye_areas_raw = eye_areas.copy()

    # pupil_areas[likely_blinks] = np.nan
    # cr_areas[likely_blinks] = np.nan
    # eye_areas[likely_blinks] = np.nan

    eye_data.insert(0, "timestamps", frame_times)
    # eye_data.insert(1, "cr_area", cr_areas)
    # eye_data.insert(2, "eye_area", eye_areas)
    # eye_data.insert(3, "pupil_area", pupil_areas)
    # eye_data.insert(4, "likely_blink", likely_blinks)
    # eye_data.insert(5, "pupil_area_raw", pupil_areas_raw)
    # eye_data.insert(6, "cr_area_raw", cr_areas_raw)
    # eye_data.insert(7, "eye_area_raw", eye_areas_raw)

    return eye_data
