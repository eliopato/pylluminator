import os.path
from enum import Enum, unique
import pandas as pd
import urllib.request
import logging
import pyranges as pr
import zipfile

from utils import column_names_to_snake_case, concatenate_non_na

LOGGER = logging.getLogger(__name__)


@unique
class Channel(Enum):
    """Probes measure either a red or green fluorescence. This class defines their names and values."""
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
    """Names of the different genome versions supported"""
    HG38 = 'hg38'
    HG19 = 'hg19'
    MM10 = 'mm10'
    MM39 = 'mm39'

    def __str__(self):
        return self.value


class ArrayType(Enum):
    """Names of the different array types supported"""
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


class GenomeInfo:
    """Additional genome information provided by external files"""

    def __init__(self, genome_version: GenomeVersion):
        """Given a genome version, load the files (cen_info.csv, cyto_band.csv, gap_info.csv, seq_length.csv and
        txns.pkl) into a GenomeVersion object. If any file is missing, set the attribute to None."""
        self.seq_length = None
        self.cen_info = None
        self.txns = None
        self.gap_info = None
        self.cyto_band = None

        if genome_version is None:
            LOGGER.warning('You must set genome version to load genome information')
            return

        folder_genome = f'./data/genomes/{genome_version}/'

        if not os.path.exists(folder_genome):
            LOGGER.warning(f'No genome information found in {folder_genome}')
            return

        # read all the csv files
        for info in ['cen_info', 'cyto_band', 'gap_info', 'seq_length', 'txns']:
            filepath = f'{folder_genome}/{info}.csv'

            # if the file is not found, check if its compressed version exists and if so, unzip it
            if not os.path.exists(filepath):
                zip_file = filepath.replace('.csv', '.zip')
                if os.path.exists(zip_file):
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(folder_genome)
                # check again, in case the unzipped file is not the right one \_o_/
                if not os.path.exists(filepath):
                    LOGGER.warning(f'Missing genome information file {filepath}.')
                    continue

            df = pd.read_csv(filepath)

            if 'End' in df.columns and 'Start' in df.columns:
                df.End = df.End.astype('int')
                df.Start = df.Start.astype('int')

            # assign the dataframe to the corresponding attribute
            self.__setattr__(info, df)

        # make gap info a pyranges object
        self.gap_info = pr.PyRanges(self.gap_info)

        # seq length is usually used as a dict
        self.seq_length = dict(zip(self.seq_length.Chromosome, self.seq_length.SeqLength))

        LOGGER.info('loading done\n')


class Annotations:
    """This class contains all the metadata associated with a certain genome version (HG39, MM10...) and array type
    (EPICv2, 450K...). The metadata includes the manifest, the mask (if any exists), and the genome information (which
    is itself a combination of several dataframes, see class GenomeInfo). Masks and Manifests are automatically
    downloaded the first time the function is called, while GenomeInfo files are already stored in the repository."""

    def __init__(self, array_type: ArrayType, genome_version: GenomeVersion):
        self.array_type = array_type
        self.genome_version = genome_version
        self.mask = self.load_annotation('mask')
        self.manifest = self.load_annotation('manifest')
        self.genome_info = GenomeInfo(genome_version)

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

    def load_annotation(self, kind: str) -> pd.DataFrame | None:
        """Download or read an annotation file. Kind must be 'mask', 'manifest', or 'genome_info'"""

        LOGGER.info(f'>> loading {kind} for {self.array_type} {self.genome_version}')

        if kind == 'genome_info':
            return GenomeInfo(self.genome_version)

        elif kind not in ['mask', 'manifest']:
            LOGGER.warning(f'Unknown annotation {kind}, must be one of `mask`, `manifest`')
            return None

        data_folder = f'./data/{kind}s/'
        tsv_filename = f'{self.array_type}.{self.genome_version}.{kind}.tsv.gz'
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

            # set dataframes index + specific processing for manifest file
            if kind == 'manifest':
                # for type I probes that have both address A and address B set, split them in two rows
                df['illumina_id'] = df.apply(lambda x: concatenate_non_na(x, ['address_a', 'address_b']), axis=1)
                df = df.explode('illumina_id', ignore_index=True)
                df['illumina_id'] = df['illumina_id'].astype('int')
                df.set_index('illumina_id', inplace=True)
                # turn some columns into categories as it speeds up further processing
                for column in ['type', 'probe_type', 'channel']:
                    df[column] = df[column].astype('category')
                # to improve readability
                df['probe_type'] = df.probe_type.cat.rename_categories({'rs': 'snp'})
            elif kind == 'mask':
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

    def get_genomic_ranges(self) -> pd.DataFrame:
        """Extract genomic ranges information from manifest dataframe"""
        genomic_ranges = self.manifest[['probe_id', 'cpg_chrm', 'cpg_beg', 'cpg_end', 'map_yd_a']].drop_duplicates()
        genomic_ranges = genomic_ranges.set_index('probe_id')
        # rename column to fit pyRanges naming convention
        genomic_ranges = genomic_ranges.rename(columns={'cpg_chrm': 'Chromosome',
                                                        'cpg_beg': 'Start',
                                                        'cpg_end': 'End',
                                                        'map_yd_a': 'Strand'})

        genomic_ranges['Strand'] = genomic_ranges.Strand.replace({'f': '-', 'r': '+', 'u': '*'}).fillna('*')
        genomic_ranges['Chromosome'] = genomic_ranges.Chromosome.fillna('*')
        genomic_ranges['Start'] = genomic_ranges.Start.fillna(0).astype(int)
        genomic_ranges['End'] = genomic_ranges.End.fillna(0).astype(int)
        return genomic_ranges

    def __str__(self):
        return f'{self.array_type} - {self.genome_version}'

    def __repr__(self):
        return f'Annotation : \nArray type : {self.array_type} - Genome version : {self.genome_version}\n'

