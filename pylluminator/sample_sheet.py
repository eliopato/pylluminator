"""Functions to generate sample sheets from idat files, or parse a provided sample sheet"""

import re
import os.path

import numpy as np
import pandas as pd
from importlib.resources.readers import MultiplexedPath

from pylluminator.utils import column_names_to_snake_case, get_files_matching, convert_to_path

from pylluminator.utils import get_logger

LOGGER = get_logger()


def read_from_file(filepath: str, delimiter: str = ',') -> pd.DataFrame | None:
    """Read sample sheet from the provided filepath. The file must be a .csv, but you can define a custom delimiter.

    Required columns in input file :
        - sample_id

    Recommended :
        - sentrix_id or sentrix_barcode or sentrix_barcode_a
        - sentrix_position or sentrix_position_a
        - sample_name: For display purposes.
        - and any other metadata you'd like to use

    Any other column will be left untouched in the sample sheet dataframe (with its name converted to snake case)

    :param filepath: path to the sample sheet.
    :type filepath: str

    :param delimiter: delimiter used in the file. Default: ','
    :type delimiter: str

    :return: the sample sheet as a dataframe, or None if the file could not be read
    :rtype: pandas.DataFrame | None
    """
    filepath = os.path.expanduser(filepath)

    extension = filepath.split('.')[-1]
    if extension != 'csv':
        LOGGER.error(f'Sample sheet file must be in .csv format, not {extension}')
        return None

    if not os.path.exists(filepath):
        LOGGER.error(f'Filepath provided for sample sheet does not exist ({filepath})')
        return None

    df = pd.read_csv(filepath, delimiter=delimiter)

    # check if the file is in the format where the header follows a [Data] line
    data_index = df.index[df.iloc[:, 0] == '[Data]']
    if len(data_index) == 1:
        df = pd.read_csv(filepath, delimiter=delimiter, skiprows=data_index[0] + 2, header=0)
    elif len(data_index) > 1:
        LOGGER.error('While reading sample sheet file : several [data] lines found in file')
        return None

    df = column_names_to_snake_case(df)
    df = df.rename(columns={'sentrixposition_a': 'sentrix_position',
                            'sentrixbarcode': 'sentrix_id', 'sentrixbarcode_a': 'sentrix_id'})

    if len(df) == 0:
        LOGGER.error(f'Empty sample sheet file {filepath}')
        return None

    if 'sample_id' not in df.columns:
        LOGGER.error(f'Column sample_id not found in {df.columns}')
        return None

    return df


def create_from_idats(idat_folder: str | os.PathLike | MultiplexedPath,
                      output_filename='samplesheet.csv') -> (pd.DataFrame, str):
    """Creates a sample sheet file from a folder containing .idat files.

    Files need to follow one of the following formats :
        - [GSM_id]_[sentrix id]_[sentrix_position]_[Grn|Red].idat
        - [sentrix id]_[sentrix_position]_[Grn|Red].idat

    :param idat_folder: path to the folder containing all the .idat files
    :type idat_folder: str |  os.PathLike | MultiplexedPath

    :param output_filename: name of the sample sheet file that will be created. Default: 'samplesheet.csv'
    :type output_filename: str

    :return: the sample_sheet created for the .idat files as a dataframe, and the filepath of the new file
    :rtype: tuple[pandas.DataFrame, str]
    """

    samples_dict = {'sample_id': [], 'sample_name': [], 'sentrix_id': [], 'sentrix_position': []}

    idat_folder = convert_to_path(idat_folder)

    if not idat_folder.exists():
        LOGGER.error(f'{idat_folder} is not a valid directory path')
        return None, None

    idat_files = get_files_matching(idat_folder, '*Grn.idat*')  # .gz OK

    for idx, idat in enumerate(idat_files):
        # split string by '/', last element is local file name
        filename = str(idat).split('/')[-1]
        split_filename = filename.split('_')

        # file name formated as [GSM_id]_[sentrix id]_[sentrix_position]_[Grn|Red].idat[.gz]
        if len(split_filename) == 4 and split_filename[0].startswith('GSM'):
            samples_dict['sentrix_id'].append(split_filename[1])
            samples_dict['sentrix_position'].append(split_filename[2])
            samples_dict['sample_id'].append(split_filename[0])
            samples_dict['sample_name'].append(split_filename[0])
        # file name formated as [sentrix id]_[sentrix_position]_[Grn|Red].idat[.gz]
        elif len(split_filename) == 3:
            samples_dict['sentrix_id'].append(split_filename[0])
            samples_dict['sentrix_position'].append(split_filename[1])
            samples_dict['sample_id'].append(f'{split_filename[0]}_{split_filename[1]}')
            samples_dict['sample_name'].append(f'{split_filename[0]}_{split_filename[1]}')
        # file name formated as [GSM_id]-bliblablou-[Grn|Red].idat[.gz]
        else:
            matched = re.match(r'(GSM\d+).(.*).?(Grn|Red)\.idat', filename)
            if matched is not None:
                samples_dict['sentrix_id'].append(np.nan)
                samples_dict['sentrix_position'].append(np.nan)
                samples_dict['sample_id'].append(matched[1])
                samples_dict['sample_name'].append(matched[2] if len(matched[2]) <= 1 else matched[2][:-1])
            else:
                LOGGER.error(f'The file {filename} does not have the right pattern to auto-generate a sample sheet. '
                             f'Please create a sample sheet file manually')

    for key in ['sentrix_id', 'sentrix_position']:
        if pd.isnull(samples_dict[key]).all():
            samples_dict.pop(key)

    df = pd.DataFrame(data=samples_dict)
    filepath = f'{idat_folder}/{output_filename}'
    df.to_csv(path_or_buf=filepath, index=False)
    LOGGER.info(f'Created sample sheet: {filepath} with {len(df)} samples')

    return df, filepath
