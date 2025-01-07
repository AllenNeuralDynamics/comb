
from pathlib import Path
import json
import sparse
import h5py
import numpy as np

# set up logger
import logging
logger = logging.getLogger(__name__)

def find_data_file(input_path, file_part, verbose=False):
    """Find a file in a directory given a partial file name.

    Example
    -------
    input_path = /root/capsule/data/multiplane-ophys_724567_2024-05-20_12-00-21
    file_part = "_sync.h5"
    return: "/root/capsule/data/multiplane-ophys_724567_2024-05-20_12-00-21/ophys/1367710111_sync.h5"
    
    
    Parameters
    ----------
    input_path : str or Path
        The path to the directory to search.
    file_part : str
        The partial file name to search for.
    """
    input_path = Path(input_path)
    try:
        file = list(input_path.glob(f'**/*{file_part}*'))[0] # Error-prone: what if there are multiple files with the same file_part?
        # TODO: add a check for multiple files with the same file_part. At least pring a warning. Best is to make sure that there is only one file.
    except IndexError:
        if verbose:
            logger.warning(f"File with '{file_part}' not found in {input_path}")
        file = None
    return file


def get_file_paths_dict(file_parts_dict, input_path):
    file_paths = {}
    for key, value in file_parts_dict.items():
        file_paths[key] = find_data_file(input_path, value)
    return file_paths


def check_ophys_folder(path):
    """ophys folders can have multiple names, check for all of them"""
    ophys_names = ['ophys', 'pophys', 'mpophys']
    ophys_folder = None
    for ophys_name in ophys_names:
        ophys_folder = path / ophys_name
        if ophys_folder.exists():
            break
        else:
            ophys_folder = None

    return ophys_folder


def check_behavior_folder(path):
    behavior_names = ['behavior', 'behavior_videos']
    behavior_folder = None
    for behavior_name in behavior_names:
        behavior_folder = path / behavior_name
        if behavior_folder.exists():
            break
        else:
            behavior_folder = None
    return behavior_folder


## update in aind-ohys-data-access
def load_signals(h5_file: Path, h5_group=None, h5_key=None) -> tuple:
    """Loads extracted signal data from aind-ophys-extraction-suite2p

    Parameters
    ----------
    h5_file: Path
    h5_group: str
        Group to access key in h5 file
    h5_key: str
        Key to extract data from
    
    Returns
    -------
    (np.array, ImageSegmentation)
        Trace array and updated segmentation object
    """
    # add warning: moved to aind-ophys-data-access, will be removed in future
    logger.warning(f"this function (load_signals) has been moved to aind-ophys-data-access, will be removed in future")
    if not h5_group:
        with h5py.File(h5_file, "r") as f:
            traces = f[h5_key][:]
    else:
        with h5py.File(h5_file, "r") as f:
            traces = f[h5_group][h5_key][:]
    index = traces.shape[0]

    roi_names = np.arange(index).tolist()

    return traces, roi_names


def load_generic_group(h5_file: Path, h5_group=None, h5_key=None) -> np.array:

    """Loads extracted signal data from aind-ophys-extraction-suite2p

    Parameters
    ----------
    h5_file: Path
        Path to h5_file
    h5_group: str
        Group to access key in h5 file
    h5_key: str
        Key to extract data from
    
    Returns
    -------
    (np.array)
        Segmentation masks on full image
    """
    # add warning: moved to aind-ophys-data-access, will be removed in future
    logger.warning(f"this function: (load_generic_group) has been moved to aind-ophys-data-access, will be removed in future")
    print("h5", h5_file)
    with h5py.File(h5_file, "r") as f:
        masks = f[h5_group][h5_key][:]
    
    return masks


def load_sparse_array(h5_file):
    # add warning: moved to aind-ophys-data-access, will be removed in future
    logger.warning(f"this function: {load_sparse_array} has been moved to aind-ophys-data-access, will be removed in future")
    with h5py.File(h5_file) as f:
        data = f["rois"]["data"][:]
        coords = f["rois"]["coords"][:]
        shape = f["rois"]["shape"][:]

    pixelmasks = sparse.COO(coords,data,shape).todense()
    return pixelmasks


def get_sync_file_path(input_path, verbose=False):
    """Find the Sync file"""
    
    # add warning: moved to aind-ophys-data-access, will be removed in future
    logger.warning(f"this function: (get_sync_file_path) has been moved to aind-ophys-data-access, will be removed in future")

    file_parts = {}
    input_path = Path(input_path)
    try: 
        # method 1: find sync_file by name
        file_parts = {"sync_h5": "_sync.h5"}
        sync_file_path = find_data_file(input_path, file_parts["sync_h5"], verbose=False)
    except IndexError as e:
        if verbose:
            logger.info("file with '*_sync.h5' not found, trying platform json")

    if sync_file_path is None:
        # method 2: load platform json
        # Note: sometimes fails if platform json has incorrect sync_file path
        logging.info(f"Trying to find sync file using platform json for {input_path}")
        file_parts = {"platform_json": "_platform.json"}
        platform_path = find_data_file(input_path, file_parts["platform_json"])
        with open(platform_path, 'r') as f:
            platform_json = json.load(f)

        ophys_folder = check_ophys_folder(input_path)
        behavior_folder = input_path / "behavior"
        
        parent_folders = [ophys_folder, behavior_folder]
        for f in parent_folders:
            if f is not None:
                try:
                    sync_file_path = f / platform_json["sync_file"]
                    if sync_file_path.exists():
                        break
                except KeyError as e:
                    sync_file_path = None
            else:
                sync_file_path = None


        if not sync_file_path.exists():
            logger.error(f"Unsupported data asset structure, sync file not found in {sync_file_path}")
            sync_file_path = None
        else:
            logger.info(f"Sync file found in {sync_file_path}")

    return sync_file_path


