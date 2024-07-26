import os.path
from enum import Enum, unique
import pandas as pd
import urllib.request

from utils import column_names_to_snake_case, concatenate_non_na


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

    def __init__(self, array_type: ArrayType, genome_version: GenomeVersion, with_mask=True):
        self.array_type = array_type
        self.genome_version = genome_version
        self.manifest = None
        self.mask = None

        # open the manifest and mask files if they already exist, or download them
        stems = ['manifest', 'mask'] if with_mask else ['manifest']
        for stem in stems:

            print(f'\n >> loading {stem} for {array_type} {genome_version}')
            filename = f'{array_type}.{genome_version}.{stem}.tsv.gz'
            filepath = f'./data/{stem}s/{filename}'

            if not os.path.exists(filepath):
                dl_link = f'https://github.com/zhou-lab/InfiniumAnnotationV1/raw/main/Anno/{array_type}/{filename}'
                print(f'file {filepath} not found, trying to download it from {dl_link}')
                try:
                    urllib.request.urlretrieve(dl_link, filepath)
                    print(f'download successful')
                except:
                    link = 'https://zwdzwd.github.io/InfiniumAnnotation'
                    print(f'download failed, you can download it from {link} and add it to the {stem} folder manually')
                    continue

            df = pd.read_csv(filepath, delimiter='\t')
            df = column_names_to_snake_case(df)
            df['probe_type'] = df['probe_id'].str.extract(r'^([a-zA-Z]+)')

            if stem == 'manifest':
                # ids to match with idat files
                # todo apply concatenate to each _a/_b column?
                df['illumina_id'] = df.apply(lambda x: concatenate_non_na(x, ['address_a', 'address_b']), axis=1)
                df = df.explode('illumina_id', ignore_index=True)
                df['illumina_id'] = df['illumina_id'].astype('int')
                df.set_index('illumina_id', inplace=True)
            elif stem == 'mask':
                df.set_index('probe_id', inplace=True)

            self.__setattr__(stem, df)
