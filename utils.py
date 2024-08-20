import pickle
import logging
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

def column_names_to_snake_case(df: pd.DataFrame) -> pd.DataFrame:
    """converts the dataframe's column names from camel case to snake case, and replace spaces by underscores"""
    # regex to detect a new word in a camel case string
    camel_case = '(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[a-zA-Z])(?=[1-9])|(?<=[1-9])(?=[A-Z])'
    # specificity, replace CpG by CPG otherwise it becomes cp_g with the regex
    df.columns = df.columns.str.replace('CpG', 'CPG')
    # convert camel case to snake case
    df.columns = df.columns.str.replace(camel_case, '_', regex=True).str.lower()
    df.columns = [c.replace(' ', '_') for c in df.columns]
    return df


def concatenate_non_na(row: pd.Series, col_names: list[str]) -> list:
    """Function to concatenate values of N columns into a list, excluding NaN"""
    values = []
    for col_name in col_names:
        if pd.notna(row[col_name]):
            values.append(row[col_name])
    return values


def get_column_as_flat_array(df: pd.DataFrame, column: str | list, remove_na: bool = False):
    """get values from one or several columns of a pandas dataframe, and return a flatten list of the values.
     If `remove_na` is set to True, all NaN values will be removed"""
    values = df[[column]].values
    if remove_na:
        return values[~np.isnan(values)]
    return values.flatten()


def mask_dataframe(df: pd.DataFrame, indexes_to_mask: pd.MultiIndex) -> pd.DataFrame:
    """Mask given indexes from the dataframe, and return the dataframe masked"""
    if indexes_to_mask is None or len(indexes_to_mask) == 0:
        return df
    return df[~df.index.isin(indexes_to_mask)]


def remove_probe_suffix(probe_id: str) -> str:
    """Remove the last part of a probe ID, split by underscore. The last part is usually 2 letters and 2 numbers,
    refering to : top or bottom strand (T/B), converted or opposite strand (C/O), Infinium probe type (1/2),
    and the number of synthesis for representation of the probe on the array (1,2,3,…,n)."""
    str_split = probe_id.split('_')
    if len(str_split) < 2:
        return probe_id
    else:
        return '_'.join(str_split[:-1])


def save_object(object_to_save, filepath: str):
    """Save any object as a pickle file"""
    LOGGER.info(f'Saving {type(object_to_save)} object in {filepath}')
    with open(filepath, 'wb') as f:
        pickle.dump(object_to_save, f)


def load_object(filepath: str, object_type):
    """Load any object from a pickle file"""
    LOGGER.info(f'Loading {object_type} object from {filepath}')
    with open(filepath, 'rb') as f:
        loaded_object = pickle.load(f)

    if not isinstance(loaded_object, object_type):
        LOGGER.error(f'The saved object type {type(loaded_object)} doesnt match the requested type ({object_type})')

    return loaded_object
