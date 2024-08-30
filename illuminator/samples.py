import os
import logging
from inspect import signature
from importlib.resources.readers import MultiplexedPath

import pandas as pd

from illuminator.sample import Sample
import illuminator.sample_sheet as sample_sheet
from illuminator.read_idat import IdatDataset
from illuminator.annotations import Annotations, Channel
from illuminator.utils import save_object, load_object, get_files_matching

LOGGER = logging.getLogger(__name__)


class Samples:
    """Samples contain a collection of Sample objects in a dictionary, with sample names as keys. It also holds the
    sample sheet information and the annotation object. It is mostly used to apply functions to several samples at a
    time"""

    def __init__(self, sample_sheet_df: pd.DataFrame | None):
        """Initialize the object with only a sample-sheet (which can be None)"""
        self.annotation = None
        self.sample_sheet = sample_sheet_df
        self.samples = {}

    def read_samples(self, datadir: str | os.PathLike | MultiplexedPath, max_samples: int | None = None) -> None:
        """Search for idat files in the datadir through all sublevels. The idat files are supposed to match the
        information from the sample sheet and follow this naming convention:
        `[sentrix ID]*[sentrix position]*[channel].idat` where the `*` can be any characters.
        `channel` must be spelled `Red` or Grn`.

        :param datadir: string or path-like pointing to the directory where sesame files are
        :param max_samples: (optional, int or None, default to None) to only load N samples to speed up the process
            (useful for testing purposes)

        :return: None"""

        LOGGER.info(f'>> start reading sample files from {datadir}')

        if self.sample_sheet is None:
            LOGGER.debug('No sample sheet provided, creating one')
            self.sample_sheet, _ = sample_sheet.create_from_idats(datadir)

        # for each sample
        for _, line in self.sample_sheet.iterrows():

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

            # add the sample to the dictionary
            self.samples[line.sample_name] = sample

            # only load the N first samples
            if max_samples is not None and len(self.samples) == max_samples:
                break

        LOGGER.info(f'reading sample files done\n')

    def merge_annotation_info(self, annotation: Annotations, light_mode=True) -> None:
        """For all samples, call the function to merge manifest and mask information to the methylation signal data
        read from idat files.

        :param annotation: Annotations object with genome version and array type corresponding to the data stored
        :param light_mode: (optional, default True) if True, load only the required columns to speed up the process

        :return: None
        """

        self.annotation = annotation
        LOGGER.info(f'>> start merging manifest and sample data frames')
        for sample in self.samples.values():
            sample.merge_annotation_info(self.annotation, light_mode)
        LOGGER.info(f'done merging manifest and sample data frames\n')

    @staticmethod
    def from_sesame(datadir: str | os.PathLike | MultiplexedPath, annotation: Annotations):
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

    def __getattr__(self, method_name):
        """Wrapper for Sample methods that can directly be applied to every sample"""

        supported_functions = ['dye_bias_correction', 'dye_bias_correction_nl', 'noob_background_correction',
                               'scrub_background_correction', 'poobah', 'infer_type1_channel', 'apply_quality_mask',
                               'apply_non_unique_mask']

        if callable(getattr(Sample, method_name)) and method_name in supported_functions:
            def method(*args, **kwargs):
                LOGGER.info(f'>> start {method_name}')
                for sample in self.samples.values():
                    getattr(sample, method_name)(*args, **kwargs)
                LOGGER.info(f'done with {method_name}\n')

            method.__name__ = method_name
            method.__doc__ = getattr(Sample, method_name).__doc__
            method.__signature__ = signature(getattr(Sample, method_name))
            return method

        LOGGER.error(f'Undefined attribute/method {method_name} for class Samples')


    ####################################################################################################################
    # Processing methods
    ####################################################################################################################

    def get_betas(self, mask: bool = True, include_out_of_band: bool = True) -> pd.DataFrame:
        """Compute beta values for all samples.
        :param mask: set to True to apply current mask to each sample.
        :param include_out_of_band: set to true to include Out-of-band signal of Type I probes in the Beta values
        :return: a dataframe with samples as column, and probes (multi-indexed) as rows """
        LOGGER.info(f'>> start calculating betas')
        betas_list = []
        for sample in self.samples.values():
            betas_list.append(sample.get_betas(mask, include_out_of_band))
        betas_df = pd.concat(betas_list, axis=1)
        LOGGER.info(f'done calculating betas\n')
        return betas_df

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
