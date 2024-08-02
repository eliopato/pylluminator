import os.path
from enum import Enum, unique
import pandas as pd
import numpy as np
import urllib.request
import logging

from utils import column_names_to_snake_case, concatenate_non_na

LOGGER = logging.getLogger(__name__)


@unique
class Channel(Enum):
    """ idat probes measure either a red or green fluorescence.
    This specifies which to return within idat.py: red_idat or green_idat."""
    RED = 'Red'
    GREEN = 'Grn'

    def __str__(self):
        return self.value

    @property
    def is_green(self):
        return self == self.GREEN

    @property
    def is_red(self):
        return self == self.RED


class GenomeVersion(Enum):
    HG38 = 'hg38'
    HG19 = 'hg19'
    MM10 = 'mm10'
    MM39 = 'mm39'

    def __str__(self):
        return self.value


class ArrayType(Enum):
    HUMAN_27K = 'HM27'
    HUMAN_450K = 'HM450'
    HUMAN_MSA = 'MSA'
    HUMAN_EPIC = 'EPIC'
    HUMAN_EPIC_PLUS = 'EPIC+'
    HUMAN_EPIC_V2 = 'EPICv2'

    MOUSE_MM285 = 'MM285'

    MAMMAL_40 = 'Mammal40'

    def __str__(self):
        return self.value


class Annotations:

    def __init__(self, array_type: ArrayType, genome_version: GenomeVersion):
        self.array_type = array_type
        self.genome_version = genome_version
        self.mask = self.load_annotation('mask')
        self.manifest = self.load_annotation('manifest')

    def download_from_github(self, data_folder: str, tsv_filename: str) -> int:
        """Download a manifest or mask from Zhou lab github page, and returns it as a dataframe. Returns -1 if the
        file could not be downloaded"""

        dl_link = f'https://github.com/zhou-lab/InfiniumAnnotationV1/raw/main/Anno/{self.array_type}/{tsv_filename}'
        LOGGER.info(f'file {tsv_filename} not found in {data_folder}, trying to download it from {dl_link}')

        os.makedirs(data_folder, exist_ok=True)  # create destination directory

        try:
            urllib.request.urlretrieve(dl_link, f'{data_folder}{tsv_filename}')
            LOGGER.info(f'download successful')
        except:
            link = 'https://zwdzwd.github.io/InfiniumAnnotation'
            LOGGER.info(f'download failed, download it from {link} and add it to the {data_folder} folder')
            return -1

        return 1

    def load_annotation(self, name: str) -> pd.DataFrame | None:
        """Download or read an annotation file. Name must be 'mask' or 'manifest'"""

        LOGGER.info(f'>> loading {name} for {self.array_type} {self.genome_version}')

        if name not in ['mask', 'manifest']:
            LOGGER.warning(f'Unknown annotation {name}, must be one of `mask`, `manifest`')
            return None

        data_folder = f'./data/{name}s/'
        tsv_filename = f'{self.array_type}.{self.genome_version}.{name}.tsv.gz'
        pkl_filename = tsv_filename.replace('tsv.gz', 'pkl')

        # if the pickled file doesn't already exist, create it
        if not os.path.exists(f'{data_folder}{pkl_filename}'):

            # if the csv manifest file doesn't exist, download it
            if not os.path.exists(f'{data_folder}{tsv_filename}'):
                if self.download_from_github(data_folder, tsv_filename) == -1:
                    return None

            # now read the downloaded manifest file
            df = pd.read_csv(f'{data_folder}{tsv_filename}', delimiter='\t')

            # uniformization - who likes camel case ?
            df = column_names_to_snake_case(df)

            # extract probe type from probe id (first letters, identifies control probes, snp...)
            df['probe_type'] = df['probe_id'].str.extract(r'^([a-zA-Z]+)')

            # set dataframes index
            if name == 'manifest':
                # for type I probes that have both address A and address B set, split them in two rows
                df['illumina_id'] = df.apply(lambda x: concatenate_non_na(x, ['address_a', 'address_b']), axis=1)
                df = df.explode('illumina_id', ignore_index=True)
                df['illumina_id'] = df['illumina_id'].astype('int')
                df.set_index('illumina_id', inplace=True)
            elif name == 'mask':
                df = df.set_index('probe_id')
                df = df.rename(columns={'mask': 'mask_info'})

            pd.to_pickle(df, f'{data_folder}{pkl_filename}')

        else:

            df = pd.read_pickle(f'{data_folder}{pkl_filename}')
            LOGGER.info('loading from pickle file done\n')

        return df

    @property
    def non_unique_mask_names(self) -> str:
        """Mask names for non-unique probes, as defined in Sesame."""
        return 'M_nonuniq|nonunique|sub35_copy|multi|design_issue'

    @property
    def quality_mask_names(self) -> str:
        """Recommended mask names for each Infinium platform, as defined in Sesame. We're assuming that EPIC+ arrays
        have the same masks as EPIC v2 arrays."""
        if self.array_type in [ArrayType.HUMAN_EPIC_V2, ArrayType.HUMAN_EPIC_PLUS]:
            names = ['M_1baseSwitchSNPcommon_5pt', 'M_2extBase_SNPcommon_5pt', 'M_mapping', 'M_nonuniq',
                     'M_SNPcommon_5pt']
        elif self.array_type in [ArrayType.HUMAN_EPIC, ArrayType.HUMAN_450K]:
            names = ['mapping', 'channel_switch', 'snp5_GMAF1p', 'extension', 'sub30_copy']
        elif self.array_type == ArrayType.HUMAN_27K:
            names = ['mask']
        elif self.array_type == ArrayType.MOUSE_MM285:
            names = ['ref_issue', 'nonunique', 'design_issue']
        else:
            LOGGER.warning(f'No quality mask names defined for array type {self.array_type}')
            names = ['']

        return '|'.join(names)

    def __str__(self):
        return f'{self.array_type} - {self.genome_version}'
