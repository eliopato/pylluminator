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

        # self.filepath = filepath
        # self.delimiter = delimiter

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


# def parse_sample_sheet_into_idat_datasets(sample_sheet, array_type: ArrayType, sample_name=None, meta_only=False, bit='float32') -> list:
#     """Generates a collection of IdatDatasets from samples in a sample sheet.
#
#     Arguments:
#         sample_sheet {SampleSheet} -- The SampleSheet from which the data originates.
#
#     Keyword Arguments:
#         sample_name {string} -- Optional: one sample to process from the sample_sheet. (default: {None})
#         meta_only {True/False} -- doesn't read idat files, only parses the meta data about them.
#         (RawMetaDataset is same as RawDataset but has no idat probe values stored in object, because not needed in pipeline)
#
#     Raises:
#         ValueError: If the number of probes between raw datasets differ.
#
#     Returns:
#         [RawDatasets] -- A list of idat data pairs, each a dict like {'green_idat': green_idat, 'red_idat': red_idat}
#     """
#     # now idat_datasets is not a class, but just a list of dicts, with each dict being a pair of red_idat and
#     # green_idat Objects.
#     if not sample_name:
#         samples = sample_sheet.get_samples()
#     elif type(sample_name) is list:
#         samples = [sample_sheet.get_sample(sample) for sample in sample_name ]
#     else:
#         samples = [sample_sheet.get_sample(sample_name)]
#         LOGGER.info("Found sample in SampleSheet: {0}".format(sample_name))
#
#     # LOGGER.info(f'Reading {len(samples)} IDATs from sample sheet')
#     if meta_only:
#         parser = RawMetaDataset
#         idat_datasets = [parser(sample) for sample in samples]
#
#     else:
#         def parser(sample):
#             green_filepath = sample.get_filepath('idat', Channel.GREEN)
#             green_idat = IdatDataset(green_filepath, channel=Channel.GREEN, bit=bit)
#             red_filepath = sample.get_filepath('idat', Channel.RED)
#             red_idat = IdatDataset(red_filepath, channel=Channel.RED, bit=bit)
#             return {'green_idat': green_idat, 'red_idat': red_idat, 'sample': sample}
#
#         idat_datasets = []
#         for sample in tqdm(samples, total=len(samples), desc='Reading IDATs'):
#             idat_datasets.append(parser(sample))
#         idat_datasets = list(idat_datasets)  # tqdm objects are not subscriptable, not like a real list
#         # ensure all idat files have same number of probes
#         batch_probe_counts = set()
#         counts_per_sample = Counter()
#         for idx, dataset in enumerate(idat_datasets):
#             snps_read = {dataset['green_idat'].n_snps_read, dataset['red_idat'].n_snps_read}
#             if len(snps_read) > 1:
#                 raise ValueError('IDAT files have a varying number of probes (compared Grn to Red channel)')
#             n_snps_read = snps_read.pop()
#             batch_probe_counts.add(n_snps_read)
#             counts_per_sample[n_snps_read] += 1
#             idat_datasets[idx]['array_type'] = array_type
#         if len(batch_probe_counts) != 1:
#             array_types = Counter([dataset['array_type'] for dataset in idat_datasets])
#             LOGGER.warning(f"These IDATs have varying numbers of probes: {counts_per_sample.most_common()} for these "
#                            f"array types: {array_types.most_common()} \n (Processing will drop any probes that are not "
#                            f"found across all samples for a given array type.)")
#     return idat_datasets