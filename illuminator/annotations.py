from enum import Enum, unique
import pandas as pd
import pyranges as pr

from illuminator.utils import get_resource_folder, get_logger, get_or_download_data

LOGGER = get_logger()
ILLUMINA_DATA_LINK = 'https://github.com/eliopato/illuminator-data/raw/main/'

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


@unique
class GenomeVersion(Enum):
    """Names of the different genome versions supported"""
    HG38 = 'hg38'
    HG19 = 'hg19'
    MM10 = 'mm10'
    MM39 = 'mm39'

    def __str__(self):
        return self.value


@unique
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

    def __init__(self, name: str, genome_version: GenomeVersion):
        """Given a genome version, load the files (gap_info.csv, seq_length.csv) into a GenomeVersion object."""

        if genome_version is None:
            LOGGER.warning('You must set genome version to load genome information')
            return

        folder_genome = get_resource_folder(f'genome_info.{name}.{genome_version}')
        dl_link = f'{ILLUMINA_DATA_LINK}/genome_info/{genome_version}/'

        # read all the csv files
        for info in ['gap_info', 'seq_length', 'chromosome_regions', 'transcripts_exons', 'transcripts_list']:

            filepath = folder_genome.joinpath(f'{info}.csv')
            if filepath.exists():
                df = pd.read_csv(str(filepath))
            elif name == 'default':
                df = get_or_download_data(folder_genome, f'{info}.csv', dl_link)
            else:
                LOGGER.error(f'File {filepath} doesn\'t exist for custom annotation info {name}, please add it')
                continue

            if df is None:
                LOGGER.error(f'not able to get {info}')
                continue

            if info == 'gap_info':
                df.End = df.End.astype('int')
                df.Start = df.Start.astype('int')
                gen_info = pr.PyRanges(df)
            elif info == 'seq_length':
                gen_info = dict(zip(df.Chromosome, df.SeqLength))
            elif info == 'transcripts_list':
                gen_info = df.set_index('group_name')
            elif info == 'transcripts_exons':
                gen_info = df.set_index('transcript_id')
            elif info == 'chromosome_regions':
                gen_info = df.set_index('Chromosome')
            else:
                gen_info = df

            self.__setattr__(info, gen_info)


class Annotations:
    """This class contains all the metadata associated with a certain genome version (HG39, MM10...) and array type
    (EPICv2, 450K...). The metadata includes the manifest, the mask (if any exists), and the genome information (which
    is itself a combination of several dataframes, see class GenomeInfo). Masks and Manifests are automatically
    downloaded the first time the function is called, while GenomeInfo files are already stored in the repository."""

    def __init__(self, array_type: ArrayType, genome_version: GenomeVersion, name='default'):
        # parameters
        self.array_type = array_type
        self.genome_version = genome_version
        self.name = name
        # load genome info files
        self.genome_info = GenomeInfo(name, genome_version)
        self.probe_infos = None
        self.genomic_ranges = None

        # load probe_info and genomic_ranges files

        data_folder = get_resource_folder(f'annotations.{self.name}.{self.genome_version}.{self.array_type}')

        for info in ['probe_infos', 'genomic_ranges']:
            df = None
            filepath = data_folder.joinpath(f'{info}.csv')

            # read or download data file
            if filepath.exists():
                LOGGER.debug(f'reading csv {filepath}')
                df = pd.read_csv(filepath, index_col=0)
            elif self.name == 'default':
                dl_link = f'{ILLUMINA_DATA_LINK}/annotations/{self.genome_version}/{self.array_type}/'
                df = get_or_download_data(data_folder, f'{info}.csv', dl_link)

            if df is None:
                LOGGER.error(f'No {info}.csv input file found for {self.name}, {self.genome_version}, {self.array_type}')
                continue

            if info == 'probe_infos':
                df[['type', 'probe_type', 'channel']] = df[['type', 'probe_type', 'channel']].astype('category')

            self.__setattr__(info, df)

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
        return f'{self.name} annotation - {self.array_type} - {self.genome_version} '

    def __repr__(self):
        return f'Annotation {self.name} : \nArray type : {self.array_type} - Genome version : {self.genome_version}\n'

