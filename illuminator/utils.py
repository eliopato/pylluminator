import os
import pickle
import tarfile
import logging
import urllib.request
from pathlib import Path, PosixPath
import zipfile
from importlib.resources import files
from importlib.resources.readers import MultiplexedPath

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


def mask_dataframe(df: pd.DataFrame, indexes_to_mask: pd.MultiIndex | list) -> pd.DataFrame:
    """Mask given indexes from the dataframe, and return the dataframe masked"""
    if indexes_to_mask is None or len(indexes_to_mask) == 0:
        return df

    if isinstance(indexes_to_mask, list):
        all_masked_indexes = set()
        for masked_indexes in indexes_to_mask:
            all_masked_indexes.update(masked_indexes)
        return df[~df.index.isin(all_masked_indexes)]

    return df[~df.index.isin(indexes_to_mask)]


def remove_probe_suffix(probe_id: str) -> str:
    """Remove the last part of a probe ID, split by underscore. The last part is usually 2 letters and 2 numbers,
    referring to : top or bottom strand (T/B), converted or opposite strand (C/O), Infinium probe type (1/2),
    and the number of synthesis for representation of the probe on the array (1,2,3,…,n).

    :param probe_id: (string) the probe name to remove the suffix from

    :return: (string) the probe name without the suffix"""
    str_split = probe_id.split('_')
    if len(str_split) < 2:
        return probe_id
    else:
        return '_'.join(str_split[:-1])


def save_object(object_to_save, filepath: str):
    """Save any object as a pickle file

    :param object_to_save: (any) any object to save as a pickle file
    :param filepath: (str) path describing where to save the file

    :return: None"""
    LOGGER.info(f'Saving {type(object_to_save)} object in {filepath}')
    with open(filepath, 'wb') as f:
        pickle.dump(object_to_save, f)


def load_object(filepath: str, object_type=None):
    """Load any object from a pickle file:

    :param filepath: (str) full path to the object to load. The file *MUST* be in pickle format
    :param object_type: (optional) type of the object, so that the function checks that it has the right type

    :return: loaded object"""

    LOGGER.info(f'Loading {object_type if object_type is not None else ""} object from {filepath}')
    with open(filepath, 'rb') as f:
        loaded_object = pickle.load(f)

    if object_type is not None:
        if not isinstance(loaded_object, object_type):
            LOGGER.error(f'The saved object type {type(loaded_object)} doesnt match the requested type ({object_type})')

    return loaded_object


def get_resource_folder(module_path: str, create_if_not_exist=True) -> MultiplexedPath | None:
    """Find the resource folder, and creates it if it doesn't exist and if the parameter is set to True (default)

    :param module_path: (str) path in a module format (e.g. "illuminator.data.genomes")
    :param create_if_not_exist: (bool, optional, default True) if the module directory doesn't exist, create it and its
        parents if necessary

    :return: path to the folder as a MultiplexedPath if it was found/created, None otherwise"""

    # check that the input module path is OK
    if not module_path.startswith('illuminator.data'):
        module_path = 'illuminator.data.' + module_path

    # find the data folder in the package
    try:
        data_folder = files(module_path)
    except (ModuleNotFoundError, NotADirectoryError):
        # the directory doesn't exist and shouldn't be created, return None
        if not create_if_not_exist:
            return None
        modules = module_path.split('.')
        if len(modules) < 2:
            LOGGER.error(f'Unable to find {module_path} data, nor create missing folders')
            return None
        parent_module = '.'.join(modules[:-1])
        child_module = modules[-1]
        # recursively check that parent directories exist
        parent_posix = get_resource_folder(parent_module)
        # once the parents exist, create the child
        parent_posix.joinpath(child_module).mkdir()
        # now the folder should exist!
        data_folder = files(module_path)

    return data_folder


def convert_to_path(input_path: str | os.PathLike | MultiplexedPath) -> Path | PosixPath:
    """Ensure the `input_path` is a PathLike format.
    If it's a string or a MultiplexedPath, convert it to a PathLike object."""
    if isinstance(input_path, MultiplexedPath):
        return input_path.joinpath('*').parent  # convert MultiplexedPath into PosixPath
    return Path(input_path)


def get_files_matching(root_path: str | os.PathLike | MultiplexedPath, pattern: str) -> list[os.PathLike]:
    """ Equivalent to Path.rglob() for MultiplexedPath. Find all files in the subtree matching the pattern"""
    return [p for p in convert_to_path(root_path).rglob(pattern)]


def download_from_geo(gsm_ids_to_download: str | list[str], target_directory: str | os.PathLike | MultiplexedPath) -> None:
    """Download idat files from GEO (Gene Expression Omnibus) given one or several GSM ids."""

    # uniformization of the input parameter : if there is only one GSM id, make it a list anyway
    if isinstance(gsm_ids_to_download, str):
        gsm_ids_to_download = [gsm_ids_to_download]

    # uniformization of the input parameter : make MultiplexedPath and strings a Path
    target_directory = convert_to_path(target_directory)
    # create the directory if it doesn't exist
    os.makedirs(target_directory, exist_ok=True)

    # download and un-tar GSM files one by one
    for gsm_id in gsm_ids_to_download:
        # check that it doesn't already exist :
        matching_files = get_files_matching(target_directory, f'{gsm_id}*idat*')
        if len(matching_files) >= 2:
            LOGGER.debug(f'idat files already exist for {gsm_id} in {target_directory}, skipping. ({matching_files}')
            continue
        # otherwise, start downloading
        LOGGER.debug(f'downloading {gsm_id}')
        target_file = f'{target_directory}/{gsm_id}.tar'
        dl_link = f'https://www.ncbi.nlm.nih.gov/geo/download/?acc={gsm_id}&format=file'
        try:
            urllib.request.urlretrieve(dl_link, target_file)
        except:
            LOGGER.error(f'download failed, download it from {dl_link} and add it to the {target_directory} folder')
            continue
        # if the download succeeded, untar the file
        with tarfile.TarFile(target_file, 'r') as tar_ref:
            tar_ref.extractall(target_directory, filter='data')
        # delete the tar file
        os.remove(target_file)

def get_chromosome_number(chromosome_id: str | list[str] | pd.Series, convert_string=False) -> list[int] | int | None:
    """From a string representing the chromosome ID, get the chromosome number. E.g. 'chr22' -> 22. If the input
    is not a numbered chromosome, return np.nan. The string part has to be only 'chr', not case-sensitive

    :param chromosome_id : (string, list of string or pd.Series) input string(s) to extract the number from
    :param convert_string : (optional, boolean, default False) convert any non-numbered chromosome to a number. Gives
        X the ID 97, Y the ID 98 and M the ID 99. Any other string will be given the ID 100

    :return: the chromosome(s) number(s) as an integer, or np.nan if not applicable"""

    # for list and series, call the function on each member
    if isinstance(chromosome_id, list) or isinstance(chromosome_id, pd.Series):
        return [get_chromosome_number(chr_id, convert_string) for chr_id in chromosome_id]

    # juste checking that it's not already an int to avoid a useless error...
    if isinstance(chromosome_id, int):
        return chromosome_id

    # remove the 'chr' part of the string to keep only the number
    trimmed_str = chromosome_id.lower().replace('chr', '')
    chromosome_number = None  # default value

    # if the remaining string only contains digits, we're good !
    if trimmed_str.isdigit():
        return int(trimmed_str)

    # if the chromosome name is a string, only convert it if the parameter is set to true
    if convert_string:
        # give letter chromosomes a number ID (they're lowercased earlier)
        supported_chrs = {'x': 97, 'y': 98, 'm': 99}
        if trimmed_str in supported_chrs.keys():
            chromosome_number = supported_chrs[trimmed_str]
        else:
            chromosome_number = 100

    return chromosome_number

def set_level_as_index(df: pd.DataFrame, level: str, drop_others=False) -> pd.DataFrame:
    """Change the index of a MultiIndexed DataFrame, to a single Index, using a level of the MultiIndex. Other levels
    will be dropped if drop_others is set to True

    :param df: (pd.DataFrame) dataframe to update
    :param level: (string) name of the index level to use as the new Index
    :param drop_others: (optional, boolean, default False) drop all the other levels (as opposed to keeping them in new
        columns)

    :return: pd.DataFrame"""
    if drop_others:
        return df.reset_index(level).reset_index(drop=True).set_index(level)
    else:
        return df.reset_index().set_index(level)

def download_from_link(dl_link: str, target_filepath: PosixPath) -> int:
    """Download a file and save it to the target. Returns -1 if the file could not be downloaded, 1 otherwise.

    :param dl_link: (string) link to the file to be downloaded
    :param target_filepath: (PosixPath) where the file will be saved

    :return: (int) exit status"""

    data_folder = target_filepath.parent
    filename = target_filepath.name

    LOGGER.debug(f'file {filename} not found in {data_folder}, trying to download it from {dl_link}')

    os.makedirs(data_folder, exist_ok=True)  # create destination directory

    try:
        urllib.request.urlretrieve(dl_link, target_filepath)
        LOGGER.debug(f'download successful')
    except:
        LOGGER.info(f'download from {dl_link} failed, try downloading it manually and save it in {data_folder}')
        return -1

    if filename.endswith('.zip'):
        LOGGER.info(f'unzip downloaded file {target_filepath}')

        with zipfile.ZipFile(target_filepath, 'r') as zip_ref:
            zip_ref.extractall(convert_to_path(data_folder))
        os.remove(target_filepath)  # remove archive

    return 1