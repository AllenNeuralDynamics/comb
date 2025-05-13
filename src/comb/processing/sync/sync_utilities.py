from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from comb import data_file_keys
import os
from comb.processing.sync.sync_dataset import SyncDataset

"""
Copied from AllenSDK (version = ) on 01/19/2023 by @mattjdavis
"""
def get_keys_for_camera_type(cam_name: str):
    '''
    Returns a tuple of line labels for a specific camera type/name
    IY 04.18.25
    '''
    cam_name_lower = cam_name.lower()
    if cam_name_lower == 'side': #side camera is named behavior
        cam_name_lower = 'behavior'
    keys = []

    for attr_name in dir(data_file_keys):
        attr_name_lower = attr_name.lower()

        if cam_name_lower in attr_name_lower:
            keys.append(attr_name)
    
    if len(keys) == 0:
        raise ValueError(f"No attribute found containing '{cam_name}' (case-insensitive)")
    elif len(keys) > 1:
        raise ValueError(f"Multiple attributes found containing '{cam_name}': {matches}")

    return getattr(data_file_keys, keys[0])


def trim_discontiguous_times(times: np.ndarray, threshold=100) -> np.ndarray:
    """
    If the time sequence is discontigous,
    detect the first instance occurance and trim off the tail of the sequence

    Parameters
    ----------
    times : frame times

    Returns
    -------
    trimmed frame times
    """

    times = np.array(times)
    intervals = np.diff(times)

    med_interval = np.median(intervals)
    interval_threshold = med_interval * threshold

    gap_indices = np.where(intervals > interval_threshold)[0]

    # A special case for when the first element is a discontiguity
    if np.abs(intervals[0]) > interval_threshold:
        gap_indices = [0]

    if len(gap_indices) == 0:
        return times

    return times[:gap_indices[0] + 1]


def get_synchronized_frame_times(session_sync_file: Path,
                                 sync_line_label_keys: Tuple[str, ...],
                                 drop_frames: Optional[List[int]] = None,
                                 trim_after_spike: bool = True,
                                 ) -> pd.Series:
    """Get experimental frame times from an experiment session sync file.

    1. Get rising edges from the sync dataset
    2. Occasionally an extra set of frame times are acquired after the rest of
        the signals. These are manifested by a discontiguous time sequence.
        We detect and remove these.
    3. Remove dropped frames

    Parameters
    ----------
    session_sync_file : Path
        Path to an ephys session sync file.
        The sync file contains rising/falling edges from a daq system which
        indicates when certain events occur (so they can be related to
        each other).
    sync_line_label_keys : Tuple[str, ...]
        Line label keys to get times for. See class attributes of
        allensdk.brain_observatory.sync_dataset.Dataset for a listing of
        possible keys.
    drop_frames : List
        frame indices to be removed from frame times
    trim_after_spike : bool = True
        If True, will call trim_discontiguous_times on the frame times
        before returning them, which will detect any spikes in the data
        and remove all elements for the list which come after the spike.

    Returns
    -------
    pd.Series
        An array of times when eye tracking frames were acquired.
    """
    sync_dataset = SyncDataset(str(session_sync_file))

    times = sync_dataset.get_edges(
        "rising", sync_line_label_keys, units="seconds"
    )
    
    sync_dataset.close() # 01/24/2025 JK

    times = trim_discontiguous_times(times) if trim_after_spike else times
    if drop_frames is not None:
        times = [t for ix, t in enumerate(times) if ix not in drop_frames]

    return pd.Series(times)

def get_total_frames(video_path: Path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get frame rate (fps) and total frame count
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return frame_count
    

def compare_and_trim(timestamps: np.ndarray, frames: np.ndarray, tag: str) -> np.ndarray:
    """
    Compare the lengths of two arrays and handle discrepancies.
    First frame is metadata, that should be dropped in video analysis.

    - If the length of `timestamps` is greater than `frames - 1`, a warning is printed, and `timestamps` is trimmed.
    - If the length of `timestamps` is less than `frames - 1`, a warning is printed, but no changes are made.
    - If the length of `timestamps` is exactly `frames - 1`, a confirmation statement is printed.

    Args:
        timestamps (np.ndarray): Array of timestamps from sync file.
        frames (np.ndarray): Array representing the number of frames in a movie.
        tag (str): Identifier for the behavior video.

    Returns:
        np.ndarray: The adjusted `timestamps` array.
    """
    len_timestamps, len_frames = len(timestamps), len(frames)

    if len_timestamps > len_frames - 1:
        print(f"Warning: Length of timestamps ({len_timestamps}) is greater than frames minus one ({len_frames - 1}). Trimming timestamps.")
        timestamps = timestamps[:len_frames - 1]
    elif len_timestamps < len_frames - 1:
        print(f"Warning: Length of timestamps ({len_timestamps}) is less than frames minus one ({len_frames - 1}). No trimming applied.")
    else:
        print(f"sync timestamps match {tag} behavior video length")
    
    return timestamps

def validate_sync_timestamps(sync_path: str, video_path: str, cam_name: str) -> np.ndarray:
    """
    Validate and align sync timestamps with video frames.

    This function extracts timestamps from the sync file based on the camera type,
    compares them with the number of frames in the corresponding video, and adjusts
    the length of the timestamps array if necessary.

    Args:
        sync_path (str): Path to the synchronization file.
        video_path (str): Path to the video file.
        cam_name (str): Camera type name (e.g., 'behavior', 'face', 'eye').

    Returns:
        np.ndarray: The validated and adjusted timestamps array.

    Notes:
        - If the length of timestamps is greater than the number of frames minus one, 
          a warning is printed, and the timestamps are trimmed to match the frame count.
        - If the length of timestamps is less than the number of frames minus one, 
          a warning is printed, but no changes are made.
        - If the length of timestamps matches the number of frames minus one, 
          a confirmation message is printed.

    Example:
        >>> sync_path = "path/to/sync_file.h5"
        >>> video_path = "path/to/video.mp4"
        >>> cam_name = "behavior"
        >>> timestamps = validate_sync_timestamps(sync_path, video_path, cam_name)
    """
    keys = get_keys_for_camera_type(cam_name)
    timestamps = get_synchronized_frame_times(sync_path, keys)
    if video_path is not None:
        frames = np.arange(get_total_frames(video_path))
        timestamps = compare_and_trim(timestamps, frames, cam_name)
    
    return timestamps


