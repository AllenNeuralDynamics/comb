# TODO: TOOK FUNCTIONS FROM MWB CAPUSLE SHOULD DELETE THERE AND ADD THESE MJD 9/19
# Copied from aind-ophys-data-access (11/14/2025 JK)
from pathlib import Path
import json
from typing import Union, Dict
import warnings
import pandas as pd

from .file_handling import check_ophys_folder

# set up logger
import logging
logger = logging.getLogger(__name__)

def extract_laser_metadata(session_json: dict):
    """Get laser metadata from the session json.

    Parameters
    ----------
    session_json: dict
        session json

    Returns
    -------
    laser_metadata: dict
    """
    ophys_stream = extract_ophys_stream(session_json)
    light_sources = ophys_stream.get('light_sources', [])
    laser_stream = next((ls for ls in light_sources if ls.get('name') == 'Laser'), None)
    if laser_stream is None:
        warnings.warn("No laser stream found in the metadata", UserWarning)
        return {
            'laser_wavelength': None,
            'wavelength_unit': None
        }
    return {
        'laser_wavelength': laser_stream.get('wavelength'),
        'wavelength_unit': laser_stream.get('wavelength_unit')
    }


def extract_ophys_stream(session_json: dict):
    """
    Extract the ophys stream from the session metadata.

    Parameters
    ----------
    metadata : dict
        The session metadata dictionary.

    Returns
    -------
    dict
        The ophys stream data.

    Raises
    ------
    ValueError
        If no ophys stream is found in the metadata.
    """
    try:
        ophys_stream = next((stream for stream in session_json['data_streams'] 
                            if any(modality.get('name') == 'Planar optical physiology' 
                                    for modality in stream.get('stream_modalities', []))), 
                            None)
    except KeyError:
        ophys_stream = {}
        warnings.warn("No ophys stream found in the metadata", UserWarning)
    
    return ophys_stream


def extract_ophys_fovs(session_json: dict,
                       index_key_only: bool = False):
    """Get a dict of ophys fovs with the index and targeted structure as the key.

    Parameters
    ----------
    session_json: dict
        session json
    index_key_only: bool
        If True, only return the index as the key.

    Returns
    -------
    structured_fovs: dict
    """
    ophys_stream = extract_ophys_stream(session_json)
    ophys_fovs = ophys_stream.get('ophys_fovs', [])
    structured_fovs = {}

    for fov in ophys_fovs:
        index = fov.get('index')
        targeted_structure = fov.get('targeted_structure')
        if isinstance(targeted_structure, dict):
            targeted_structure = targeted_structure.get('acronym')

        if index is not None and targeted_structure is not None:
            
            if index_key_only:
                key = index
            else:
                key = f"{targeted_structure}_{index}"
            structured_fovs[key] = fov

    return structured_fovs

def cre_driver_from_genotype(genotype: str):
    """Get the cre driver line from the genotype.

    Parameters
    ----------
    genotype: str
        The genotype of the subject.

    Returns
    -------
    cre_driver: str
    """
    try:
        genotype_parts = genotype.split('-')
        cre_driver = genotype_parts[0]

        # Check for wildtype genotype
        if genotype.lower() in ['wt/wt', 'wt/wt ']:
            warnings.warn("Wildtype genotype detected. "
                        "Viral injection for Cre driver not implemented.",
                        UserWarning)
            cre_driver = None
    except AttributeError:
        return None

    return cre_driver

def gcamp_from_genotype(genotype: str):
    """Get the gcamp version from the genotype.

    Parameters
    ----------
    genotype: str
        The genotype of the subject.

    Returns
    -------
    gcamp_version: str
    """
    try:
        genotype_parts = genotype.split('-')
        gcamp_part = [part for part in genotype_parts if 'gcamp' in part.lower() or 'gc' in part.lower()]

        # Check for wildtype genotype
        if genotype.lower() in ['wt/wt', 'wt/wt ']:
            warnings.warn("Wildtype genotype detected. "
                        "Viral injection for GCaMP not implemented.",
                        UserWarning)
            return None
    except AttributeError:
        return None

    if gcamp_part:
        return gcamp_part[0]
    else:
        return None
    
# from pandas import json_normalize

# def parse_procedures(procedures: dict):
#     """Parse the procedures json to get the task description.

#     Parameters
#     ----------
#     procedures: dict
#         The procedures json.

#     Returns
#     -------
#     task_description: str
#     """
    
#     sp = procedures.get('subject_procedures')
    
#     if sp is None:
#         # warning older version of procedures.json
#         logger.warning("No subject_procedures found in procedures.json. May return missing metadata.")
#         return None
        
#     main_level_df = json_normalize(
#         sp,
#         errors='ignore',
#         sep='_'
#     )
#     main_level_df = main_level_df.rename(columns={'procedure_type': 'main_procedure_type'})

#     procedures_df = main_level_df.explode('procedures')
    
    
#     # procedures is dict, find procedure_type and put in main level
#     # procedures_df = procedures_df.join(
#     #     json_normalize(procedures_df.pop('procedures'))
#     # )
    
#     return procedures_df


def metadata_for_multiplane_session(record: dict, docdb_record: bool = False) -> dict:
    """Extract metadata to build nwb ophys file

    Parameters
    ----------
    data_path: dict
        record from doc sb 

    Returns
    -------
    metadata: dict

    """

    md = {}

    if docdb_record:
        md['asset_name'] = record['name']
        md['asset_id'] = record['_id']
        md['asset_location'] = record['location']
    
    json_keys = ["session", "subject", "data_description", "rig", "procedures"]
    for key in json_keys:
        if record[key] is None:
            record[key] = {}
            logger.warning(f"{key} not found for {md['asset_name']}")

    ### SESSION METADATA ###
    # 2024-05-17T08:35:11.779675-07:00
    md['session_start_time'] = record['session'].get('session_start_time')
    md['session_date'] = md['session_start_time'].split('T')[0]
    md['experimenter'] = record['session'].get('experimenter_full_name')
    md['session_type'] = record['session'].get('session_type')
    md.update(extract_laser_metadata(record['session']))
    md['ophys_fovs'] = extract_ophys_fovs(record['session'])

    for fov in md['ophys_fovs'].values():
        if isinstance(fov.get('targeted_structure'), dict):
            fov['targeted_structure'] = fov['targeted_structure'].get('acronym')

    md['session_targeted_structures'] = list(set(
        fov['targeted_structure'] 
        for fov in md['ophys_fovs'].values()
        if 'targeted_structure' in fov
    ))

    ### SESSION SUMMARY METADATA ###
    md['session_num_planes'] = len(md['ophys_fovs'].keys())

    imaging_depths = []
    for fov in md['ophys_fovs'].values():
        if 'imaging_depth' in fov:
            imaging_depths.append(fov['imaging_depth'])

    md['session_imaging_depths'] = sorted(imaging_depths)

    # This should be a natural text summary of the task
    md['session_task_description'] = "N/A"

    ### RIG METADATA ###
    # could get from rig, but also in session.json
    md['microscope_name'] = record['session'].get('rig_id')
    if md['microscope_name'] in ["MESO.1", "MESO.2"]:
        md['microscope_description'] = "AIND Multiplane Mesoscope 2P Rig"
    else:
        md['microscope_description'] = "N/A"

    ### SUBJECT METADATA ###
    md['subject_id'] = record['subject'].get('subject_id')
    md['genotype'] = record['subject'].get('genotype')
    md['sex'] = record['subject'].get('sex')
    md['date_of_birth'] = record['subject'].get('date_of_birth')
    md['gcamp'] = gcamp_from_genotype(md['genotype'])
    md['cre_driver'] = cre_driver_from_genotype(md['genotype'])
    md["session_key"] = md["subject_id"] + "_" + md["session_date"]

    # TODO: better metadata for plane segmentation
    md['ophys_seg_approach'] = "N/A" #"Cellpose"-
    md['ophys_seg_descr'] = "N/A" # "Cellpose segmentation of two-photon movie"

    # TODO: procedure parse and add date since surgery/virus

    return md

#######
# Loading direct from Jsons
#######

def load_metadata_json_files(session_path: Union[str, Path],asset_type="raw") -> Dict[str, Union[dict, None]]:
    """
    Load procedures.json, session.json, subject.json, and rig.json into a dictionary.

    Parameters
    ----------
    session_path : Union[str, Path]
        Path to the session directory.
        Can be raw or processed (raw greatly preferred)

    Returns
    -------
    Dict[str, Union[dict, None]]
        A dictionary containing the loaded JSON data for each file.
        If a file is not found, its value will be None.
    """
    session_path = Path(session_path)
    json_files = ['procedures.json', 'session.json', 'subject.json', 'rig.json',
                  'data_description.json']
    result = {}

    for file_name in json_files:
        file_path = session_path / file_name
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    result[file_name.replace('.json', '')] = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding {file_name}. File may be empty or contain invalid JSON.")
                result[file_name.replace('.json', '')] = {}
        else:
            print(f"File {file_name} not found in {session_path}")
            result[file_name.replace('.json', '')] = {}
    
    # "*_platform.json only in raw assets
    # mesoscope json on in v4 pipeline ouputs
    if asset_type =="raw":
        ophys_folder = check_ophys_folder(session_path)
        
        # "*_platform.json" find in ophys folder with glob
        platform_json = list(ophys_folder.glob('*_platform.json'))
        if len(platform_json) > 0:
            with open(platform_json[0], 'r') as f:
                result['platform'] = json.load(f)
        else:
            result['platform'] = None

        meso_split_json = list(ophys_folder.glob('MESOSCOPE_FILE_SPLITTING_QUEUE*'))
        if len(meso_split_json) > 0:
            with open(meso_split_json[0], 'r') as f:
                result['mesoscope_splitting_json'] = json.load(f)
        else:
            result['mesoscope_splitting_json'] = None

    return result

# this function can look at jsons directly, but now we prefer
# to use docdb
# def metadata_for_multiplane_OLD(data_path: Union[str, Path]) -> dict:
#     """Extract metadata to build nwb ophys file


#     Parameters
#     ----------
#     data_path: Union[str, Path]
#         path to the processed/raw multiplane ophys session folder

#     Returns
#     -------
#     metadata: dict

#     """

#     md = {}

#     jsons = load_metadata_json_files(data_path)

#     ### SESSION METADATA ###
#     md['session_start_time'] = jsons['session'].get('session_start_time')
#     md['experimenter'] = jsons['session'].get('experimenter_full_name')
#     md['session_type'] = jsons['session'].get('session_type')
#     md.update(extract_laser_metadata(jsons['session']))
#     md['ophys_fovs'] = extract_ophys_fovs(jsons['session'])

#     md['session_targeted_structures'] = list(set(
#         fov['targeted_structure'] 
#         for fov in md['ophys_fovs'].values()
#         if 'targeted_structure' in fov
#     ))

#     ### SESSION SUMMARY METADATA ###
#     md['session_num_planes'] = len(md['ophys_fovs'].keys())

#     imaging_depths = []
#     for fov in md['ophys_fovs'].values():
#         if 'imaging_depth' in fov:
#             imaging_depths.append(fov['imaging_depth'])

#     md['session_imaging_depths'] = sorted(imaging_depths)

#     # This should be a natural text summary of the task
#     md['session_task_description'] = "Visual change detection task."

#     ### RIG METADATA ###
#     # could get from rig, but also in session.json
#     md['microscope_name'] = jsons['session'].get('rig_id')
#     if md['microscope_name'] in ["MESO.1", "MESO.2"]:
#         md['microscope_description'] = "AIND Multiplane Mesoscope 2P Rig"
#     else:
#         md['microscope_description'] = "Unknown"

#     ### SUBJECT METADATA ###
#     md['subject_id'] = jsons['subject'].get('subject_id')
#     md['genotype'] = jsons['subject'].get('genotype')
#     md['sex'] = jsons['subject'].get('sex')
#     md['gcamp'] = gcamp_from_genotype(md['genotype'])

#     # TODO: better metadata for plane segmentation
#     md['ophys_seg_approach'] = "Cellpose"
#     md['ophys_seg_descr'] = "Cellpose segmentation of two-photon movie"

#     return md