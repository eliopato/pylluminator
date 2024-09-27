from enum import Enum, unique
import pandas as pd
import pyranges as pr

from illuminator.utils import get_resource_folder, download_from_link, column_names_to_snake_case, concatenate_non_na
from illuminator.utils import get_logger

LOGGER = get_logger()


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

        module_path = f'annotations.{name}.{genome_version}.genome_info'
        folder_genome = get_resource_folder(module_path, create_if_not_exist=False)
        if folder_genome is None:
            LOGGER.warning(f'No genome information found for {name} {genome_version}')
            return

        # read all the csv files
        for info in ['gap_info', 'seq_length']:
            filepath = str(folder_genome.joinpath(f'{info}.csv'))

            df = pd.read_csv(filepath)

            if info == 'gap_info':
                df.End = df.End.astype('int')
                df.Start = df.Start.astype('int')
                self.gap_info = pr.PyRanges(df)
            else:
                self.seq_length = dict(zip(df.Chromosome, df.SeqLength))


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
        # annotations
        self.genome_info = GenomeInfo(name, genome_version)
        self.probe_infos = None
        self.genomic_ranges = None
        self.get_or_make_illuminator_annotations()  # get probe info and genomic ranges

    def get_or_make_illuminator_annotations(self) -> None:
        """Get probe_info and genomic_ranges objects from data files or create them automatically from Sesame data"""
        data_folder = get_resource_folder(f'annotations.{self.name}.{self.genome_version}.{self.array_type}')
        not_found = False

        # try reading files first
        for info in ['probe_infos', 'genomic_ranges']:
            filepath = data_folder.joinpath(f'{info}.csv')
            if filepath.exists():
                LOGGER.info(f'reading csv {filepath}')
                df = pd.read_csv(filepath, index_col=0)
                if info == 'probe_infos':
                    df[['type', 'probe_type', 'channel']] = df[['type', 'probe_type', 'channel']].astype('category')
                self.__setattr__(info, df)
            elif self.name == 'default':
                not_found = True
            else:
                LOGGER.error(f'No {info}.csv input file found for {self.name}, {self.genome_version}, {self.array_type}')

        # if any file was not found, and it's not a custom array, create files from Sesame Data
        if not_found:

            LOGGER.info('Illuminator annotation not found, creating it from sesame')

            # read sesame annotation files
            sesame_annotations = SesameAnnotations(self.array_type, self.genome_version)

            # make probe info dataframe
            if self.probe_infos is None:
                self.probe_infos = sesame_annotations.make_illuminator_probe_info()
                if self.probe_infos is not None:
                    self.probe_infos.to_csv(data_folder.joinpath(f'probe_infos.csv'))

            # make genomic ranges
            if self.genomic_ranges is None:
                self.genomic_ranges = sesame_annotations.genomic_ranges
                if self.genomic_ranges is not None:
                    self.genomic_ranges.to_csv(data_folder.joinpath(f'genomic_ranges.csv'))


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


class SesameAnnotations:
    """Extract meaningful information from Sesame data files, and create dataframes with Illuminator format"""

    def __init__(self, array_type: ArrayType, genome_version: GenomeVersion):
        self.array_type = array_type
        self.genome_version = genome_version
        self.mask = self.load_annotation('mask')
        self.manifest = self.load_annotation('manifest')
        self.genome_info = self.load_annotation('genome_info')
        self.gene = self.load_annotation('gene')
        self.genomic_ranges = self.make_genomic_ranges()

    def load_annotation(self, kind: str) -> pd.DataFrame | None:
        """Download or read an annotation file. Kind must be 'mask', 'manifest', 'genome_info' or 'gene'"""

        LOGGER.debug(f'>> loading {kind} for {self.array_type} {self.genome_version} from Sesame')

        # genome info files are handled separately
        if kind == 'genome_info':
            return GenomeInfo('default', self.genome_version)

        # now we can handle mask and manifest files, check that the parameter is not something else
        if kind not in ['mask', 'manifest', 'gene']:
            LOGGER.warning(f'Unknown annotation {kind}, must be one of `mask`, `manifest`, `gene`')
            return None

        # get the annotation resource folder
        data_folder = get_resource_folder('tmp')

        # build the filenames depending on the array type and genome version
        if kind == 'gene':
            # gene files have sub-versions ..
            gene_file_version = ''
            if self.genome_version == GenomeVersion.HG38:
                if self.array_type in [ArrayType.HUMAN_MSA, ArrayType.HUMAN_EPIC_V2]:
                    gene_file_version = 'v41'
                elif self.array_type in [ArrayType.HUMAN_450K, ArrayType.HUMAN_27K, ArrayType.HUMAN_EPIC]:
                    gene_file_version = 'v36'
            elif self.genome_version == GenomeVersion.HG19:
                if self.array_type in [ArrayType.HUMAN_EPIC_V2, ArrayType.HUMAN_450K, ArrayType.HUMAN_27K, ArrayType.HUMAN_EPIC]:
                    gene_file_version = 'v26lift37'
            elif self.genome_version == GenomeVersion.MM10 and self.array_type == ArrayType.MOUSE_MM285:
                gene_file_version = 'vM25'
            if gene_file_version == '':
                LOGGER.warning(f'Gene information : unsupported version {self.genome_version} / {self.array_type}')
                return None
            filename = f'{self.array_type}.{self.genome_version}.manifest.gencode.{gene_file_version}.tsv.gz'
        else:
            filename = f'{self.array_type}.{self.genome_version}.{kind}.tsv.gz'

        # file path specificity for mouse array version MM10
        if self.genome_version == 'mm10':
            filename = f'N296070/{filename}'

        filepath = data_folder.joinpath(filename)

        # if the csv manifest file doesn't exist, download it from sesame repository
        if not filepath.exists():
            dl_link = f'https://github.com/zhou-lab/InfiniumAnnotationV1/raw/main/Anno/{self.array_type}/{filename}'
            if download_from_link(dl_link, filepath) == -1:
                return None

        # now read the downloaded manifest file
        df = pd.read_csv(str(filepath), delimiter='\t')

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
            df = df.set_index('illumina_id')
            # turn some columns into categories as it speeds up further processing
            df = df.rename(columns={'design_type': 'type'}, errors='ignore') # for older manifest versions
            df[['type', 'probe_type', 'channel']] = df[['type', 'probe_type', 'channel']].astype('category')
            # to improve readability
            df['probe_type'] = df.probe_type.cat.rename_categories({'rs': 'snp'})
            df['cpg_loc'] = df['cpg_beg'] + (df['cpg_end'] - df['cpg_beg']) / 2
        else:
            df = df.set_index('probe_id')
            if kind == 'mask':
                df = df.rename(columns={'mask': 'mask_info'})

        LOGGER.info('loading done\n')
        return df

    def make_genomic_ranges(self) -> pd.DataFrame | None:
        """Extract genomic ranges information from manifest dataframe"""
        # rename column to fit pyRanges naming convention
        if self.manifest is None:
            LOGGER.warning('Make genomic ranges : provide a manifest first')
            return None

        genomic_ranges = self.manifest.rename(columns={'cpg_chrm': 'Chromosome', 'cpg_beg': 'Start', 'cpg_end': 'End',
                                                       'map_yd_a': 'Strand', 'probe_strand': 'Strand'}, errors='ignore')
        genomic_ranges = genomic_ranges[['probe_id', 'Chromosome', 'Start', 'End', 'Strand']].drop_duplicates()
        genomic_ranges = genomic_ranges.set_index('probe_id')


        genomic_ranges['Strand'] = genomic_ranges.Strand.replace({'f': '-', 'r': '+', 'u': '*'}).fillna('*')
        genomic_ranges['Chromosome'] = genomic_ranges.Chromosome.fillna('*')
        genomic_ranges['Start'] = genomic_ranges.Start.fillna(0).astype(int)
        genomic_ranges['End'] = genomic_ranges.End.fillna(0).astype(int)
        return genomic_ranges

    def make_illuminator_probe_info(self) -> pd.DataFrame | None:
        """Extract useful information from Sesame Manifest, Masks and Genes annotation and merge it in one dataframe
        :return: a pd.DataFrame with IlluminaID as indexes, probes as rows and probes info as columns"""

        if self.manifest is None:
            LOGGER.warning('Make illuminator probe info : provide a manifest first')
            return None

        # select manifest column
        manifest = self.manifest[['probe_id', 'type', 'probe_type', 'channel', 'address_a', 'address_b', 'cpg_loc']]

        if self.mask is not None:
            # select mask column (`mask_uniq` or column `mask_info` to get all  the information)
            mask = self.mask[['mask_uniq']].rename(columns={'mask_uniq': 'mask_info'})
            mask.mask_info = mask.mask_info.str.replace(',', ';').replace('"', '')
            manifest = manifest.join(mask, on='probe_id')

        if self.gene is not None:
            # select genes columns
            genes = self.gene[['genes_uniq', 'transcript_types']].rename(columns={'genes_uniq': 'genes'})
            manifest = manifest.join(genes, on='probe_id')
            manifest.transcript_types = manifest.transcript_types.apply(lambda x: ';'.join(set(str(x).replace('nan', '').split(';'))))

        return manifest