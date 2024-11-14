"""
The Annotations module provides classes and methods to handle genome metadata : Illumina manifest with probes information,
Genome version (hg38, mm10...), Array types (EPIC, MSA...), Channels (Red/Green).

The default annotation data is read from illuminator-data package, but you can add your own annotations.
"""
from enum import Enum, unique
import pandas as pd
import pyranges as pr

from illuminator.utils import get_resource_folder, get_logger, get_or_download_data

LOGGER = get_logger()
ILLUMINA_DATA_LINK = 'https://github.com/eliopato/illuminator-data/raw/main/'

@unique
class Channel(Enum):
    """Probes measure either a red or green fluorescence. This class defines their names and values."""
    RED = 'Red' #: red channel
    GREEN = 'Grn'  #: green channel

    def __str__(self):
        return self.value

    @property
    def is_green(self) -> bool:
        """Check if the current channel is green.

        :return: True if channel is green, False otherwise
        :rtype: bool"""
        return self == self.GREEN

    @property
    def is_red(self) -> bool:
        """Check if the current channel is red.

        :return: True if channel is red, False otherwise
        :rtype: bool"""
        return self == self.RED


@unique
class GenomeVersion(Enum):
    """Names of the different genome versions supported"""
    HG38 = 'hg38'  #: Human Genome, build 38
    HG19 = 'hg19'  #: Human Genome, build 19
    MM10 = 'mm10'  #: Mouse (Mus Musculus) Genome, build 10
    MM39 = 'mm39'  #: Mouse (Mus Musculus) Genome, build 39

    def __str__(self):
        return self.value


@unique
class ArrayType(Enum):
    """Names of the different array types supported"""
    HUMAN_27K = 'HM27'  #: Human Methylation 27K, CpG sites
    HUMAN_450K = 'HM450'  #: Human Methylation 450K, CpG sites
    HUMAN_MSA = 'MSA'  #: Human Methylation MSA (>450K CpG sites)
    HUMAN_EPIC = 'EPIC'  #: Human Methylation EPIC (around 850K CpG sites)
    HUMAN_EPIC_PLUS = 'EPIC+'  #: Human Methylation EPIC+ (around 850K CpG sites + double coverage of some probes for qc)
    HUMAN_EPIC_V2 = 'EPICv2'  #: Human Methylation EPIC V2 (around 950K CpG sites)

    MOUSE_MM285 = 'MM285'  #: Mouse Methylation, 285K CpG sites

    MAMMAL_40 = 'Mammal40'  #: Mammalian array, 40K CpG sites

    def __str__(self):
        return self.value


class   GenomeInfo:
    """Additional genome information provided by external files, downloaded from illumina-data.

    :ivar gap_info: contains information on gaps in the genomic sequence. These gaps represent regions that are not
        sequenced or are known to be problematic in the data, such as areas that may have low coverage or
        difficult-to-sequence regions.
    :vartype gap_info: pyranges.PyRanges

    :ivar seq_length: keys are chromosome identifiers (e.g., chr1, chrX, etc.), and the values are the corresponding
        sequence lengths (in base pairs)
    :vartype seq_length: dict

    :ivar transcripts_list: high-level overview of the transcripts and their boundaries (start and end positions)
    :vartype transcripts_list: pandas.DataFrame

    :ivar transcripts_exons: information at the level of individual exons within each transcript (type, gene name, gene id...)
    :vartype transcripts_exons: pandas.DataFrame

    :ivar chromosome_regions: Names, adresses and Giemsa stain pattern of all chromosomes' regions
    :vartype chromosome_regions: pandas.DataFrame
    """

    def __init__(self, name: str, genome_version: GenomeVersion):
        """Load the files corresponding to the given genome version, and structure the information.

        :param name: Name of the genome you want to load. Set to 'default' for Illumina default version, otherwise must
            correspond to the folder name containing you custom data
        :type name: str
        :param genome_version: genome version to load (hg32, mm10...)
        :type genome_version: GenomeVersion"""

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
    downloaded the first time the function is called, while GenomeInfo files are already stored in the repository.

    :ivar array_type: Illumina array type (EPIC, MM285...)
    :vartype arrau_type: ArrayType

    :ivar genome_version: version of the genome (HG38, MM10...)
    :vartype genome_version: GenomeVersion

    :ivar name: name of the annotation: default for illuminator-data annotations, or the name of your custom data.
    :vartype name: str

    :ivar genome_info: genome metadata for the given genome version
    :vartype genome_info: GenomeInfo

    :ivar probe_infos: probes metadata (aka Manifest), contains the probes type, address, channel, mask info...
    :vartype probe_infos: pandas.DataFrame

    :ivar genomic_ranges:
    :vartype genomic_ranges: pyranges.PyRanges
    """

    def __init__(self, array_type: ArrayType, genome_version: GenomeVersion, name='default'):
        """Get annotation corresponding to the array type and genome version

        :param array_type: illumina array type (EPIC, MSA...)
        :type array_type: ArrayType
        :param genome_version: genome version to load (hg32, mm10...)
        :type genome_version: GenomeVersion
        :param name: Name of the genome you want to load. Set to 'default' for Illumina default version, otherwise must
            correspond to the folder name containing you custom data
        :type name: str
        """
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
        """Mask names for non-unique probes, as defined in Sesame.

        :return: mask names, each name separated by a |
        :rtype: str"""
        return 'M_nonuniq|nonunique|sub35_copy|multi|design_issue'

    @property
    def quality_mask_names(self) -> str:
        """Recommended mask names for each Infinium platform, as defined in Sesame. We're assuming that EPIC+ arrays
        have the same masks as EPIC v2 arrays.

        :return: mask names, each name separated by a |
        :rtype: str"""
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

