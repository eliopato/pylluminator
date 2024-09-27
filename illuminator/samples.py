import os
from inspect import signature

from importlib.resources.readers import MultiplexedPath

import pandas as pd

from illuminator.sample import Sample
import illuminator.sample_sheet as sample_sheet
from illuminator.read_idat import IdatDataset
from illuminator.annotations import Annotations, Channel
from illuminator.utils import save_object, load_object, get_files_matching, mask_dataframe, get_logger

LOGGER = get_logger()


class Samples:
    """Samples contain a collection of Sample objects in a dictionary, with sample names as keys. It also holds the
    sample sheet information and the annotation object. It is mostly used to apply functions to several samples at a
    time"""

    def __init__(self, sample_sheet_df: pd.DataFrame | None = None):
        """Initialize the object with only a sample-sheet (which can be None)"""
        self.annotation = None
        self.sample_sheet = sample_sheet_df
        self.samples = {}

    def __getitem__(self, item: int | str):
        if isinstance(item, int):
            return self.samples[list(self.samples.keys())[item]]
        return self.samples[item]

    def keys(self):
        return self.samples.keys()

    def __getattr__(self, method_name):
        """Wrapper for Sample methods that can directly be applied to every sample"""

        supported_functions = ['dye_bias_correction', 'dye_bias_correction_nl', 'noob_background_correction',
                               'scrub_background_correction', 'poobah', 'infer_type1_channel', 'apply_quality_mask',
                               'apply_non_unique_mask', 'merge_annotation_info', 'calculate_betas']

        if callable(getattr(Sample, method_name)) and method_name in supported_functions:
            def method(*args, **kwargs):
                LOGGER.info(f'>> start {method_name}')
                [getattr(sample, method_name)(*args, **kwargs) for sample in self.samples.values()]

                # if the method called updated the beta values, update the dataframe
                if method_name == 'calculate_betas':
                    LOGGER.info(f'concatenating betas dataframes')
                    self._betas_df = pd.concat([sample.betas(False) for sample in self.samples.values()], axis=1)
                elif method_name not in ['apply_quality_mask', 'apply_non_unique_mask']:
                    self._betas_df = None

                LOGGER.info(f'done with {method_name}\n')

            method.__name__ = method_name
            method.__doc__ = getattr(Sample, method_name).__doc__
            method.__signature__ = signature(getattr(Sample, method_name))
            return method

        LOGGER.error(f'Undefined attribute/method {method_name} for class Samples')

    def betas(self, mask: bool = True):
        if mask:
            masked_indexes = [sample.masked_indexes for sample in self.samples.values()]
            return mask_dataframe(self._betas_df, masked_indexes)
        else:
            return self._betas_df

    ####################################################################################################################
    # Properties
    ####################################################################################################################

    @property
    def nb_samples(self) -> int:
        """Count the number of samples contained in the object
        :return: int"""
        return len(self.samples)

    ####################################################################################################################
    # Description, saving & loading
    ####################################################################################################################

    def __str__(self):
        return f'{[sample_name for sample_name in self.samples.keys()]}'

    def __repr__(self):
        description = f'<{self.__module__}.{type(self).__name__} object at {hex(id(self))}>\n'
        description += '\n=====================================================================\n'
        description += 'Samples object :\n'
        description += '=====================================================================\n'

        description += 'No annotation\n' if self.annotation is None else self.annotation.__repr__()
        description += '\n---------------------------------------------------------------------\n'

        description += 'No sample' if self.samples is None else f'{self.nb_samples} sample(s) :\n{self.__str__()}'
        description += '\n---------------------------------------------------------------------\n'

        description += 'No sample sheet' if self.sample_sheet is None else (f'Sample sheet first items : \n '
                                                                            f'{self.sample_sheet.head(3)}')
        description += '\n=====================================================================\n'
        return description

    def save(self, filepath: str) -> None:
        """Save the current Samples object to `filepath`, as a pickle file

        :return: None"""
        save_object(self, filepath)

    @staticmethod
    def load(filepath: str):
        """Load a pickled Samples object from `filepath`

        :return: the loaded object"""
        return load_object(filepath, Samples)


def read_samples(datadir: str | os.PathLike | MultiplexedPath,
                 sample_sheet_df: pd.DataFrame | None = None,
                 sample_sheet_name: str | None = None,
                 annotation: Annotations | None = None,
                 max_samples: int | None = None,
                 min_beads=1) -> Samples | None:
    """Search for idat files in the datadir through all sublevels. The idat files are supposed to match the
    information from the sample sheet and follow this naming convention:
    `[sentrix ID]*[sentrix position]*[channel].idat` where the `*` can be any characters.
    `channel` must be spelled `Red` or Grn`.

    :param datadir: (string or path-like) pointing to the directory where sesame files are
    :param sample_sheet_df: (optional, pd.DataFrame) samples information. If not given, will be automatically rendered
    :param sample_sheet_name: (optional, string) name of the csv file containing the samples' information. You cannot
        provide both a sample sheet dataframe and name.
    :param annotation: (optional, Annotations) probes information
    :param max_samples: (optional, int or None, default to None) to only load N samples to speed up the process
        (useful for testing purposes)
    :param min_beads: (optional, int, default to 1) filter probes that have less than N beads

    :return: Samples object or None if an error was raised"""

    LOGGER.info(f'>> start reading sample files from {datadir}')

    if sample_sheet_df is not None and sample_sheet_name is not None:
        LOGGER.error('You can\'t provide both a sample sheet dataframe and name. Please only provide one parameter.')
        return None
    elif sample_sheet_df is None and sample_sheet_name is None:
        LOGGER.debug('No sample sheet provided, creating one')
        sample_sheet_df, _ = sample_sheet.create_from_idats(datadir)
    elif sample_sheet_name is not None:
        sample_sheet_df = sample_sheet.read_from_file(f'{datadir}/{sample_sheet_name}')

    # check that the sample sheet was correctly created / read. If not, abort mission.
    if sample_sheet_df is None:
        return None

    samples_dict = {}

    # for each sample
    for _, line in sample_sheet_df.iterrows():

        sample = Sample(line.sample_name)

        # read each channel file
        for channel in Channel:
            pattern = f'*{line.sentrix_id}*{line.sentrix_position}*{channel}*.idat*'
            paths = [str(p) for p in get_files_matching(datadir, pattern)]
            if len(paths) == 0:
                LOGGER.error(f'no paths found matching {pattern}')
                continue
            if len(paths) > 1:
                LOGGER.error(f'Too many files found matching {pattern} : {paths}')
                continue
            LOGGER.debug(f'reading file {paths[0]}')
            # set the sample's idata for this channel
            sample.set_idata(channel, IdatDataset(paths[0]))

        if sample.idata is None:
            LOGGER.error(f'no idat files found for sample {line.sample_name}, skipping it')
            continue

        # add the sample to the dictionary
        samples_dict[line.sample_name] = sample

        # only load the N first samples
        if max_samples is not None and len(samples_dict) == max_samples:
            break

    samples = Samples(sample_sheet_df)
    samples.samples = samples_dict

    if annotation is not None:
        samples.annotation = annotation
        samples.merge_annotation_info(annotation, min_beads)

    LOGGER.info(f'reading sample files done\n')
    return samples


def from_sesame(datadir: str | os.PathLike | MultiplexedPath, annotation: Annotations) -> Samples:
    """Reads all .csv files in the directory provided, supposing they are SigDF from SeSAMe saved as csv files.

    :param datadir: string or path-like pointing to the directory where sesame files are
    :param annotation: Annotations object with genome version and array type corresponding to the data stored

    :return: a Samples object"""
    LOGGER.info(f'>> start reading sesame files')
    samples = Samples(None)
    samples.annotation = annotation

    # fin all .csv files in the subtree depending on datadir type
    file_list = get_files_matching(datadir, '*.csv')

    # load all samples
    for csv_file in file_list:
        sample = Sample.from_sesame(csv_file, annotation)
        samples.samples[sample.name] = sample

    LOGGER.info(f'done reading sesame files\n')
    return samples