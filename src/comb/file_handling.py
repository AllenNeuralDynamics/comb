
from pathlib import Path
import json

# set up logger
import logging
logger = logging.getLogger(__name__)

def find_data_file(input_path, file_part):
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
        file = list(input_path.glob(f'**/*{file_part}*'))[0]
    except IndexError:
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


def get_sync_file_path(input_path):
    """Find the Sync file"""
    file_parts = {}
    input_path = Path(input_path)
    try: 
        # method 1: find sync_file by name
        file_parts = {"sync_h5": "_sync.h5"}
        sync_file_path = find_data_file(input_path, file_parts["sync_h5"])
    except IndexError as e:
        logger.info("file with '*_sync.h5' no found, trying platform json")

    if sync_file_path is None:
        # method 2: load platform json
        # Note: sometimes fails if platform json has incorrect sync_file path
        logging.info(f"Trying to find sync file using platform json for {input_path}")
        file_parts = {"platform_json": "_platform.json"}
        platform_path = find_data_file(input_path, file_parts["platform_json"])
        with open(platform_path, 'r') as f:
            platform_json = json.load(f)
        ophys_folder = check_ophys_folder(input_path)
        sync_file_path = ophys_folder / platform_json['sync_file']

        if not sync_file_path.exists():
            logger.error(f"Unsupported data asset structure, sync file not found in {sync_file_path}")
            sync_file_path = None
        else:
            logger.info(f"Sync file found in {sync_file_path}")

    return sync_file_path


