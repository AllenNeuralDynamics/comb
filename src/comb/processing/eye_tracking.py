from pathlib import Path

import numpy as np
import pandas as pd

from scipy import ndimage, stats


def load_eye_tracking_hdf(eye_tracking_file: Path) -> pd.DataFrame:
    """Load a DeepLabCut hdf5 file containing eye tracking data into a
    dataframe.

    Note: The eye tracking hdf5 file contains 3 separate dataframes. One for
    corneal reflection (cr), eye, and pupil ellipse fits. This function
    loads and returns this data as a single dataframe.

    Parameters
    ----------
    eye_tracking_file : Path
        Path to an hdf5 file produced by the DeepLabCut eye tracking pipeline.
        The hdf5 file will contain the following keys: "cr", "eye", "pupil".
        Each key has an associated dataframe with the following
        columns: "center_x", "center_y", "height", "width", "phi".

    Returns
    -------
    pd.DataFrame
        A dataframe containing combined corneal reflection (cr), eyelid (eye),
        and pupil data. Column names for each field will be renamed by
        prepending the field name. (e.g. center_x -> eye_center_x)
    """
    eye_tracking_fields = ["cr", "eye", "pupil"]

    eye_tracking_dfs = []
    for field_name in eye_tracking_fields:
        field_data = pd.read_hdf(eye_tracking_file, key=field_name)

        # new aind-capsule outputs some columsn with names already
        rename_cols = [col for col in field_data.columns if not col.startswith(field_name)]
        new_col_name_map = {col_name: f"{field_name}_{col_name}"
                            for col_name in rename_cols}
        field_data.rename(new_col_name_map, axis=1, inplace=True)
        eye_tracking_dfs.append(field_data)

    eye_tracking_data = pd.concat(eye_tracking_dfs, axis=1)
    eye_tracking_data.index.name = 'frame'

    # Values in the hdf5 may be complex (likely an artifact of the ellipse
    # fitting process). Take only the real component.
    eye_tracking_data = eye_tracking_data.apply(lambda x: np.real(x.to_numpy()))  # noqa: E501

    return eye_tracking_data.astype(float)