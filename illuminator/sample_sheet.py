import os.path
import pandas as pd
import logging
from importlib.resources.readers import MultiplexedPath

from illuminator.utils import column_names_to_snake_case, get_files_matching, convert_to_path

LOGGER = logging.getLogger(__name__)


def read_from_file(filepath: str = '', delimiter: str = ',') -> pd.DataFrame | None:
    """Read sample sheet from the provided filepath. You can define a delimiter.
    Required columns in input file :
    - sentrix_id or sentrix_barcode or sentrix_barcode_a
    - sentrix_position or sentrix_position_a
    Optional :
    - sample_name : name the sample. If non-existent, the name is the value of sentrix_id
    Any other column will be left untouched in the sample sheet dataframe (with its name converted to snake case)"""

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

    # check that we have the 3 required columns
    if 'sample_name' not in df.columns:
        if 'sample_id' in df.columns:
            df['sample_name'] = df['sample_id']
            LOGGER.info(f'Column sample_name not found in {df.columns}, taking name from column sample_id')
        else:
            LOGGER.error(f'Column sample_name not found in {df.columns}')
            return None

    if 'sentrix_id' not in df.columns:
        LOGGER.error(f'Column sentrix_id or sentrix_barcode not found in {df.columns}')
        return None

    if 'sentrix_position' not in df.columns:
        LOGGER.error(f'Column sentrix_position not found in {df.columns}')
        return None

    return df


def create_from_idats(idat_folder: str | os.PathLike | MultiplexedPath,
                      output_filename='samplesheet.csv', sample_type: str = '') -> (pd.DataFrame, str):
    """Creates a samplesheet.csv file from a folder containing .IDAT files. Files need to follow a format :
    either [GSM_id]_[sentrix id]_[sentrix_position]_[Grn|Red].idat
    or [sentrix id]_[sentrix_position]_[Grn|Red].idat"""

    samples_dict = {'GSM_ID': [], 'sample_name': [], 'sentrix_id': [], 'sentrix_position': [], 'sample_type': []}

    idat_folder = convert_to_path(idat_folder)
    if not idat_folder.exists():
        raise FileNotFoundError(f'{idat_folder} is not a valid directory path')

    idat_files = get_files_matching(idat_folder, '*Grn.idat*')  # .gz OK

    for idx, idat in enumerate(idat_files):
        # split string by '/', last element is local file name
        filename = str(idat).split('/')[-1]
        split_filename = filename.split('_')

        # file name formated as [GSM_id]_[sentrix id]_[sentrix_position]_[Grn|Red].idat
        if len(split_filename) == 4 and split_filename[0].startswith('GSM'):
            samples_dict['GSM_ID'].append(split_filename[0])
            samples_dict['sentrix_id'].append(split_filename[1])
            samples_dict['sentrix_position'].append(split_filename[2])
        # file name formated as [sentrix id]_[sentrix_position]_[Grn|Red].idat
        elif len(split_filename) == 3:
            samples_dict['GSM_ID'].append('')
            samples_dict['sentrix_id'].append(split_filename[0])
            samples_dict['sentrix_position'].append(split_filename[1])
        else:
            LOGGER.error(f'The file {filename} does not have the right pattern to auto-generate a sample sheet.')

        samples_dict['sample_type'].append(sample_type)
        samples_dict['sample_name'].append(f'Sample_{idx}')

    df = pd.DataFrame(data=samples_dict)
    filepath = f'{idat_folder}/{output_filename}'
    df.to_csv(path_or_buf=filepath, index=False)
    LOGGER.info(f'Created sample sheet: {filepath} with {len(df)} samples')

    return df, filepath
