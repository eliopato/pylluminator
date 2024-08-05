import os.path
import pandas as pd
import logging

from utils import column_names_to_snake_case

LOGGER = logging.getLogger(__name__)


class SampleSheet:

    def __init__(self, filepath: str, delimiter: str):

        extension = filepath.split('.')[-1]
        if extension != 'csv':
            LOGGER.error(f'Sample sheet file must be in .csv format, not {extension}')
            return

        if not os.path.exists(filepath):
            LOGGER.error(f'Filepath provided for sample sheet does not exist ({filepath})')
            return

        self.df = None
        self.samples = dict()

        df = pd.read_csv(filepath, delimiter=delimiter)

        # check if the file is in the format where the header follows a [Data] line
        data_index = df.index[df.iloc[:, 0] == '[Data]']
        if len(data_index) == 1:
            df = pd.read_csv(filepath, delimiter=delimiter, skiprows=data_index[0]+2, header=0)
        elif len(data_index) > 1:
            LOGGER.error('While reading sample sheet file : several [data] lines found in file')
            return

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
                return

        if 'sentrix_id' not in df.columns:
            LOGGER.error(f'Column sentrix_id or sentrix_barcode not found in {df.columns}')
            return

        if 'sentrix_position' not in df.columns:
            LOGGER.error(f'Column sentrix_position not found in {df.columns}')
            return

        self.df = df
