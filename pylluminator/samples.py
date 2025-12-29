"""Class that holds a collection of Sample objects and defines methods for batch processing."""
import os
import re
import gc
from importlib.resources.readers import MultiplexedPath
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF as ecdf
from inmoose.pycombat import pycombat_norm

import pylluminator.sample_sheet as sample_sheet
from pylluminator.stats import norm_exp_convolution, quantile_normalization_using_target, background_correction_noob_fit
from pylluminator.stats import iqr
from pylluminator.utils import get_column_as_flat_array, set_channel_index_as, remove_probe_suffix, merge_dataframe_by, get_chromosome_number
from pylluminator.utils import save_object, load_object, get_files_matching, get_logger, convert_to_path, merge_alt_chromosomes
from pylluminator.read_idat import IdatDataset
from pylluminator.annotations import Annotations, Channel, ArrayType, detect_array, GenomeVersion
from pylluminator.mask import MaskCollection, Mask

LOGGER = get_logger()

class Samples:
    """
     Samples objects hold sample methylation signal in a dataframe, as well as annotation information, sample sheet data and probes masks.

    :ivar annotation: probes metadata. Default: None.
    :vartype annotation: Annotations | None
    :ivar sample_sheet: samples information given by the csv sample sheet. Default: None
    :vartype sample_sheet: pandas.DataFrame | None
    :ivar min_beads: minimum number of beads required for a probe to be considered. Default: None
    :vartype min_beads: int | None
    :ivar idata: dictionary of dataframes containing the raw signal values for each sample and channel. Default: {}
    :vartype idata: dict[str, dict[Channel, pandas.DataFrame]]
    :ivar masks: collection of probes masks. Default: MaskCollection()
    :vartype masks: MaskCollection
    """

    def __init__(self, sample_sheet_df: pd.DataFrame | None = None):
        """Initialize the object with only a sample-sheet.

        :param sample_sheet_df: sample sheet dataframe. Default: None
        :type sample_sheet_df: pandas.DataFrame | None"""
        self.annotation = None
        self.sample_sheet = sample_sheet_df
        self.min_beads = None
        self.idata = {}
        self.masks = MaskCollection()
        self._signal_df = None
        self._betas = None

    def __getitem__(self, item: int | str | list[str]) -> pd.DataFrame | None:
        if self._signal_df is not None:
            if isinstance(item, int) and item < len(self.sample_labels):
                return self._signal_df[[self.sample_labels[0]]].copy()
            elif isinstance(item, str) and item in self.sample_labels:
                return self._signal_df[[item]].copy()
            elif isinstance(item, list):
                samples_names = []
                for sample in item:
                    if sample not in self.sample_labels:
                        LOGGER.warning(f'Could not find sample "{sample}" in {self.sample_labels}')
                    else:
                        samples_names.append(sample)
                if len(samples_names) > 0:
                    return self._signal_df[samples_names].copy()
            LOGGER.error(f'Could not find item {item} in {self.sample_labels} of length {self.nb_samples}')
        else:
            LOGGER.error('No signal dataframe')
        return None

    ####################################################################################################################
    # Properties
    ####################################################################################################################

    @property
    def sample_labels(self) -> list[str]:
        """Return the names of the samples contained in this object, that also exist in the sample sheet.

        :return: the list of names
        :rtype: list[str]"""
        level_name_ini = self.sample_label_name
        samples_signal_df =  set(self._signal_df.columns.get_level_values(0))
        if level_name_ini not in self.sample_sheet.columns:
            LOGGER.warning(f'Signal dataframe level {level_name_ini} not found in sample sheet columns')
            return list(samples_signal_df)
        samples_in_sheet = set(self.sample_sheet[level_name_ini].values.tolist())
        return list(samples_in_sheet & samples_signal_df)

    @property
    def sample_label_name(self) -> str:
        """Return the name of the sample sheet column used as sample labels. By default, sample_name is used when
        creating the signal dataframe, but it can be changed by using the function merge_samples_by

        :return: the name of the identifier
        :rtype: str"""
        return self._signal_df.columns.names[0]

    @property
    def nb_samples(self) -> int:
        """Count the number of samples contained in the object

        :return: number of samples
        :rtype: int"""
        return len(self.sample_labels)

    @property
    def nb_probes(self) -> int:
        """Count the number of probes in the signal dataframe

        :return: number of probes
        :rtype: int"""
        return len(self._signal_df)

    @property
    def probe_ids(self) -> list[str]:
        """Return the list of probe IDs contained in the signal dataframe

        :return: list of probe IDs
        :rtype: list[str]"""
        return self._signal_df.index.get_level_values('probe_id').tolist()

    def type1(self, apply_mask: bool = True, sigdf:pd.DataFrame| None=None) -> pd.DataFrame:
        """Get the subset of Infinium type I probes, and apply the mask if `apply_mask` is True

        :param apply_mask: True removes masked probes, False keeps them. Ignored if sigdf is provided. Default: True
        :type apply_mask: bool

        :param sigdf: signal dataframe to use. Useful to save time applying the mask. Default: None
        :type sigdf: pd.DataFrame | None

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        if sigdf is None:
            sigdf =  self.get_signal_df(apply_mask)

        return sigdf.xs('I', level='type', drop_level=False)

    def type2(self, apply_mask: bool = True, sigdf:pd.DataFrame| None=None) -> pd.DataFrame:
        """Get the subset of Infinium type II probes, and apply the mask if `apply_mask` is True

        :param apply_mask: True removes masked probes, False keeps them. Ignored if sigdf is provided. Default: True
        :type apply_mask: bool

        :param sigdf: signal dataframe to use. Useful to save time applying the mask. Default: None
        :type sigdf: pd.DataFrame | None

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        if sigdf is None:
            sigdf =  self.get_signal_df(apply_mask)
        type_ii_df = sigdf.xs('II', level='type', drop_level=False)  # get only type II probes
        type_ii_df = type_ii_df.loc[:, type_ii_df.columns.get_level_values('signal_channel').isin(['R', 'G'])]
        return type_ii_df.dropna(axis=1, how='all')

    def oob(self, apply_mask: bool=True, sigdf:pd.DataFrame| None=None) -> pd.DataFrame | None:
        """Get the subset of out-of-band probes (for type I probes only), and apply the mask if `apply_mask` is True

        :param apply_mask: True removes masked probes, False keeps them. Ignored if sigdf is provided. Default: True
        :type apply_mask: bool

        :param sigdf: signal dataframe to use. Useful to save time applying the mask. Default: None
        :type sigdf: pd.DataFrame | None

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame | None
        """
        if sigdf is None:
            sigdf =  self.get_signal_df(apply_mask)
        return pd.concat([self.oob_green(sigdf=sigdf), self.oob_red(sigdf=sigdf)]).sort_index(axis=1)

    def oob_red(self, apply_mask: bool = True, sigdf:pd.DataFrame| None=None) -> pd.DataFrame:
        """Get the subset of out-of-band red probes (for type I probes only), and apply the mask if `apply_mask` is True

        :param apply_mask: True removes masked probes, False keeps them. Ignored if sigdf is provided. Default: True
        :type apply_mask: bool

        :param sigdf: signal dataframe to use. Useful to save time applying the mask. Default: None
        :type sigdf: pd.DataFrame | None

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        if sigdf is None:
            sigdf =  self.get_signal_df(apply_mask)
        green_probes = sigdf.xs('G', level='channel', drop_level=False)
        return green_probes.loc[:, (slice(None), 'R')]

    def oob_green(self, apply_mask: bool = True, sigdf:pd.DataFrame| None=None) -> pd.DataFrame:
        """Get the subset of out-of-band green probes (for type I probes only), and apply the mask if `apply_mask` is True

        :param apply_mask: True removes masked probes, False keeps them. Ignored if sigdf is provided. Default: True
        :type apply_mask: bool

        :param sigdf: signal dataframe to use. Useful to save time applying the mask. Default: None
        :type sigdf: pd.DataFrame | None

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        if sigdf is None:
            sigdf =  self.get_signal_df(apply_mask)
        red_probes = sigdf.xs('R', level='channel', drop_level=False)
        return red_probes.loc[:, (slice(None), 'G')]

    def ib_red(self, apply_mask: bool = True, sigdf:pd.DataFrame| None=None) -> pd.DataFrame:
        """Get the subset of in-band red probes (for type I probes only), and apply the mask if `apply_mask` is True

        :param apply_mask: True removes masked probes, False keeps them. Ignored if sigdf is provided. Default: True
        :type apply_mask: bool

        :param sigdf: signal dataframe to use. Useful to save time applying the mask. Default: None
        :type sigdf: pd.DataFrame | None

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        if sigdf is None:
            sigdf =  self.get_signal_df(apply_mask)
        red_probes = sigdf.xs('R', level='channel', drop_level=False)
        return red_probes.loc[:, (slice(None), 'R')]

    def ib_green(self, apply_mask: bool = True, sigdf:pd.DataFrame| None=None) -> pd.DataFrame:
        """Get the subset of in-band green probes (for type I probes only), and apply the mask if `apply_mask` is True

        :param apply_mask: True removes masked probes, False keeps them. Ignored if sigdf is provided. Default: True
        :type apply_mask: bool

        :param sigdf: signal dataframe to use. Useful to save time applying the mask. Default: None
        :type sigdf: pd.DataFrame | None

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        if sigdf is None:
            sigdf =  self.get_signal_df(apply_mask)
        green_probes = sigdf.xs('G', level='channel', drop_level=False)
        return green_probes.loc[:, (slice(None), 'G')]

    def ib(self, apply_mask: bool = True, sigdf:pd.DataFrame| None=None) -> pd.DataFrame:
        """Get the subset of in-band probes (for type I probes only), and apply the mask if `apply_mask` is True

        :param apply_mask: True removes masked probes, False keeps them. Ignored if sigdf is provided. Default: True
        :type apply_mask: bool

        :param sigdf: signal dataframe to use. Useful to save time applying the mask. Default: None
        :type sigdf: pd.DataFrame | None

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        if sigdf is None:
            sigdf =  self.get_signal_df(apply_mask)
        return pd.concat([self.ib_red(sigdf=sigdf), self.ib_green(sigdf=sigdf)]).sort_index(axis=1)

    def type1_green(self, apply_mask: bool = True, sigdf:pd.DataFrame| None=None) -> pd.DataFrame:
        """Get the subset of type I green probes, and apply the mask if `apply_mask` is True

        :param apply_mask: True removes masked probes, False keeps them. Ignored if sigdf is provided. Default: True
        :type apply_mask: bool

        :param sigdf: signal dataframe to use. Useful to save time applying the mask. Default: None
        :type sigdf: pd.DataFrame | None

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        if sigdf is None:
            sigdf =  self.get_signal_df(apply_mask)
        return sigdf.xs( 'G', level='channel', drop_level=False)

    def type1_red(self, apply_mask: bool = True, sigdf:pd.DataFrame| None=None) -> pd.DataFrame:
        """Get the subset of type I red probes, and apply the mask if `apply_mask` is True

        :param apply_mask: True removes masked probes, False keeps them. Ignored if sigdf is provided. Default: True
        :type apply_mask: bool

        :param sigdf: signal dataframe to use. Useful to save time applying the mask. Default: None
        :type sigdf: pd.DataFrame | None

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        if sigdf is None:
            sigdf =  self.get_signal_df(apply_mask)
        return sigdf.xs('R', level='channel', drop_level=False)

    def meth(self, apply_mask: bool = True, sigdf:pd.DataFrame| None=None) -> pd.DataFrame:
        """Get the subset of methylated probes, and apply the mask if `apply_mask` is True

        :param apply_mask: True removes masked probes, False keeps them. Ignored if sigdf is provided. Default: True
        :type apply_mask: bool

        :param sigdf: signal dataframe to use. Useful to save time applying the mask. Default: None
        :type sigdf: pd.DataFrame | None

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        if sigdf is None:
            sigdf =  self.get_signal_df(apply_mask)
        return sigdf.xs('M', level='methylation_state', drop_level=False, axis=1)

    def unmeth(self, apply_mask: bool = True, sigdf:pd.DataFrame| None=None) -> pd.DataFrame:
        """Get the subset of unmethylated probes, and apply the mask if `apply_mask` is True

        :param apply_mask: True removes masked probes, False keeps them. Ignored if sigdf is provided. Default: True
        :type apply_mask: bool

        :param sigdf: signal dataframe to use. Useful to save time applying the mask. Default: None
        :type sigdf: pd.DataFrame | None

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        if sigdf is None:
            sigdf =  self.get_signal_df(apply_mask)
        return sigdf.xs('U', level='methylation_state', drop_level=False, axis=1)

    def cg_probes(self, apply_mask: bool = True, sigdf:pd.DataFrame| None=None) -> pd.DataFrame:
        """Get CG (CpG) type probes, and apply the mask if `apply_mask` is True

        :param apply_mask: True removes masked probes, False keeps them. Ignored if sigdf is provided. Default: True
        :type apply_mask: bool

        :param sigdf: signal dataframe to use. Useful to save time applying the mask. Default: None
        :type sigdf: pd.DataFrame | None

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        return self.get_probes_with_probe_type('cg', apply_mask, sigdf)

    def ch_probes(self, apply_mask: bool = True, sigdf:pd.DataFrame| None=None) -> pd.DataFrame:
        """Get CH (CpH) type probes, and apply the mask if `apply_mask` is True

        :param apply_mask: True removes masked probes, False keeps them. Ignored if sigdf is provided. Default: True
        :type apply_mask: bool

        :param sigdf: signal dataframe to use. Useful to save time applying the mask. Default: None
        :type sigdf: pd.DataFrame | None

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        return self.get_probes_with_probe_type('ch', apply_mask, sigdf)

    def snp_probes(self, apply_mask: bool = True, sigdf:pd.DataFrame| None=None) -> pd.DataFrame:
        """Get SNP type probes ('rs' probes in manifest, but replaced by 'snp' when loaded), and apply the mask if
        `apply_mask` is True

        :param apply_mask: True removes masked probes, False keeps them. Ignored if sigdf is provided. Default: True
        :type apply_mask: bool

        :param sigdf: signal dataframe to use. Useful to save time applying the mask. Default: None
        :type sigdf: pd.DataFrame | None

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        return self.get_probes_with_probe_type('snp', apply_mask, sigdf)

    def get_probes_with_probe_type(self, probe_type: str, apply_mask: bool = True, sigdf:pd.DataFrame| None=None) -> pd.DataFrame:
        """Select probes by probe type, meaning e.g. CG, Control, SNP... (not infinium type I/II type), and apply the
        mask if `apply_mask` is True

        :param probe_type: the type of probe to select (e.g. 'cg', 'snp'...)
        :type probe_type: str

        :param apply_mask: True removes masked probes, False keeps them. Ignored if sigdf is provided. Default: True
        :type apply_mask: bool

        :param sigdf: signal dataframe to use. Useful to save time applying the mask. Default: None
        :type sigdf: pd.DataFrame | None

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        if probe_type not in self._signal_df.index.get_level_values('probe_type'):
            LOGGER.warning(f'no {probe_type} probes found')
            return pd.DataFrame()

        if sigdf is None:
            sigdf =  self.get_signal_df(apply_mask)

        return sigdf.xs(probe_type, level='probe_type', drop_level=False)

    def get_probes(self, probe_ids: list[str] | str, apply_mask: bool = True, sigdf:pd.DataFrame| None=None) -> pd.DataFrame:
        """Returns the probes dataframe filtered on a list of probe IDs

        :param probe_ids: the IDs of the probes to select
        :type probe_ids: list[str]

        :param apply_mask: True removes masked probes, False keeps them. Ignored if sigdf is provided. Default: True
        :type apply_mask: bool

        :param sigdf: signal dataframe to use. Useful to save time applying the mask. Default: None
        :type sigdf: pd.DataFrame | None

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        if probe_ids is None or len(probe_ids) == 0:
            return pd.DataFrame()

        if isinstance(probe_ids, str):
            probe_ids = [probe_ids]

        if sigdf is None:
            sigdf =  self.get_signal_df(apply_mask)

        return sigdf[sigdf.index.get_level_values('probe_id').isin(probe_ids)]


    ####################################################################################################################
    # Description, saving & loading
    ####################################################################################################################

    def __str__(self):
        return ", ".join(self.sample_labels)

    def __repr__(self):
        description = f'Samples object with {self.nb_samples} samples: {", ".join(self.sample_labels)}\n'
        description += 'No annotation\n' if self.annotation is None else self.annotation.__repr__()
        description += f'{len(self._signal_df):,} probes'
        return description

    def copy(self):
        """Create a copy of the Samples object

        :return: a copy of the object
        :rtype: Samples"""
        new_samples = Samples()
        new_samples.sample_sheet = self.sample_sheet.copy()
        new_samples.annotation = self.annotation.copy()
        new_samples.min_beads = self.min_beads
        new_samples.idata = None if self.idata is None else self.idata.copy()
        new_samples.masks =  None if self.masks is None else self.masks.copy()
        new_samples._signal_df = None if self._signal_df is None else self._signal_df.copy()
        new_samples._betas = None if self._betas is None else self._betas.copy()
        return new_samples

    def save(self, filepath: str) -> None:
        """Save the current Samples object to `filepath`, as a pickle file

        :param filepath: path to the file to create
        :type filepath: str

        :return: None"""
        save_object(self, filepath)

    @staticmethod
    def load(filepath: str) -> 'Samples':
        """Load a pickled Samples object from `filepath`

        :param filepath: path to the file to read
        :type filepath: str

        :return: the loaded object"""
        return load_object(filepath, Samples)

    ####################################################################################################################
    # Data loading
    ####################################################################################################################

    def add_annotation_info(self, annotation: Annotations, label_column: str, keep_idat=False, min_beads=1) -> None:
        """Merge manifest dataframe with probe signal values read from idat files to build the signal dataframe, adding
        channel information, methylation state and mask names for each probe.

        For manifest file, merging is done on `Illumina IDs`, contained in columns `address_a` and `address_b` of the
        manifest file.

        :param annotation: annotation data corresponding to the sample
        :type annotation: Annotations
        :param label_column: the name of the sample sheet column used for sample labels (eg sample_id, sample_name)
        :type label_column: str
        :param min_beads: filter probes with less than min_beads beads. Default: 1
        :type min_beads: int
        :param keep_idat: if set to True, keep idat data after merging the annotations. Default: False
        :type keep_idat: bool

        :return: None"""

        self.min_beads = min_beads

        # select probes signal values from idat dataframe, filtering by the minimum number of beads required
        probe_df_list = []
        for sample_label, channel_dict in self.idata.items():
            sample_dfs = []
            for channel, channel_df in channel_dict.items():
                df = channel_df.copy()
                df.loc[df.n_beads < min_beads, 'mean_value'] = pd.NA
                df['channel'] = str(channel)[0]
                df = df[['channel', 'mean_value']]
                df = df.reset_index().set_index(['illumina_id', 'channel'])
                df.columns = [sample_label]
                sample_dfs.append(df)
            probe_df_list.append(pd.concat(sample_dfs))

        probe_df = pd.concat(probe_df_list, axis=1)

        # as this function is memory-intensive, free memory when we can
        del probe_df_list
        if not keep_idat:
            del self.idata
            self.idata = None
        gc.collect()

        # auto detect annotation if not provided
        if annotation is None:
            probe_count = len(probe_df)  // 2 # nb of rows in the df = nb of rows in the idat file = nb of probes
            array_type = detect_array(probe_count)
            if array_type.is_human():
                annotation = Annotations(array_type, genome_version=GenomeVersion.HG38)
            else:
                annotation = Annotations(array_type, genome_version=GenomeVersion.MM39)
            LOGGER.info(f'Annotation auto-detected to be {annotation.array_type} - genome version set to {annotation.genome_version}')

        self.annotation = annotation

        # prepare dataframes for merge
        indexes = ['type', 'channel', 'probe_type', 'probe_id', 'mask_info']
        probe_info = annotation.probe_infos[indexes + ['address_a', 'address_b']]
        probe_df = probe_df.reset_index().rename(columns={'channel': 'signal_channel'}).drop_duplicates()
        nb_probes_before_merge = len(probe_df)
        sample_df = pd.merge(probe_df, probe_info, how='inner', on='illumina_id')
        del probe_df # free memory
        gc.collect()

        # check the number of probes lost in the merge
        lost_probes = nb_probes_before_merge - len(sample_df)
        pct_lost = 100 * lost_probes / nb_probes_before_merge
        LOGGER.info(f'Lost {lost_probes:,} illumina probes ({pct_lost:.2f}%) while merging information with Manifest')

        # deduce methylation state (M = methylated, U = unmethylated) depending on infinium type
        sample_df['methylation_state'] = '?'
        sample_df.loc[(sample_df['type'] == 'II') & (sample_df.signal_channel == 'G'), 'methylation_state'] = 'M'
        sample_df.loc[(sample_df['type'] == 'II') & (sample_df.signal_channel == 'R'), 'methylation_state'] = 'U'
        sample_df.loc[(sample_df['type'] == 'I') & (sample_df.illumina_id == sample_df.address_b), 'methylation_state'] = 'M'
        sample_df.loc[(sample_df['type'] == 'I') & (sample_df.illumina_id == sample_df.address_a), 'methylation_state'] = 'U'
        # remove probes in unknown state (missing information in manifest)
        # nb_unknown_states = sum(sample_df.methylation_state == '?')
        nb_unknown_states = sample_df.methylation_state.value_counts().get('?', 0)
        if nb_unknown_states > 0:
            LOGGER.info(f'Dropping {nb_unknown_states} probes with unknown methylation state')
            sample_df = sample_df[sample_df.methylation_state != '?']

        # drop columns that we don't need anymore
        sample_df = sample_df.drop(columns=['address_a', 'address_b', 'illumina_id'])

        # reshape dataframe to have something resembling sesame data structure - one row per probe
        sample_df = sample_df.pivot(index=indexes, columns=['signal_channel', 'methylation_state'])

        # index column 'channel' corresponds by default to the manifest channel. But it could change by calling
        # 'infer_type_i_channel()' e.g., so we need to keep track of the manifest_channel in another column
        # self._signal_df['manifest_channel'] = self._signal_df.index.get_level_values('channel').values

        # make mask_info a column, not an index - and set NaN values to empty string to allow string search on it
        sample_df['mask_info'] = sample_df.index.get_level_values('mask_info').fillna('').values
        sample_df = sample_df.reset_index(level='mask_info', drop=True)
        sample_df = sample_df.sort_index(axis=1)
        sample_df.columns = sample_df.columns.rename(label_column, level=0)

        self._signal_df = sample_df

        for sample_label in self.sample_labels:
            has_na = sample_df[[(sample_label, 'G', 'M'), (sample_label, 'R', 'U')]].isna().any(axis=1)
            self.masks.add_mask(Mask(f'min_beads_{min_beads}', sample_label, has_na.index[has_na]))


    ####################################################################################################################
    # Properties & getters
    ####################################################################################################################

    def get_signal_df(self, apply_mask: bool = True) -> pd.DataFrame:
        """Get the methylation signal dataframe, and apply the mask if `apply_mask` is True

        :param apply_mask: True set masked probes values to None. Default: True
        :type apply_mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        sigdf = self._signal_df.copy()

        if not isinstance(apply_mask, bool):
            LOGGER.warning('Mask should be a boolean, setting it to True')
            apply_mask = True

        if apply_mask:

            # apply masks sample per sample (each time, the masks common to all samples are included)
            for sample_label in self.sample_labels:
                sample_mask = self.masks.get_mask(sample_label=sample_label)
                if sample_mask is not None and len(sample_mask) > 0:
                    sample_mask = sample_mask[sample_mask.isin(sigdf.index)]
                    if len(sample_mask) > 0:
                        sigdf.loc[sample_mask, sample_label] = None

        return sigdf

    def merge_samples_by(self, by: str, apply_mask=True) -> None:
        """Merge the beads signal values of different samples by averaging them. Modifies the signal dataframe directly
        and removes p values column since their values need to be updated. Beta values are averaged as not to lose
        the batch correction result if needed. Masks are reset - but masked probes values are ignored if `apply_mask` is
        True

        :param by: the column name in the sample sheet to group samples by
        :type by: str

        :param apply_mask: skip masked probes values when merging samples if True. Default: True
        :type apply_mask: bool
        """
        if by not in self.sample_sheet.columns:
            LOGGER.error(f'Column {by} not found in sample sheet in columns')
            return

        level_name_ini = self.sample_label_name

        if by == level_name_ini:
            LOGGER.info(f'samples are already merged by {by}, nothing to do')
            return

        self.reset_poobah()  # remove p_values columns
        column_names_ini = self._signal_df.columns.get_level_values(0)
        sheet = self.sample_sheet[self.sample_sheet[level_name_ini].isin(column_names_ini)]

        signal_df =  self.get_signal_df(apply_mask=apply_mask)
        beta_df = self.get_betas(apply_mask=apply_mask)

        signal_df_list = {}
        beta_df_list = {}
        for new_name, group in sheet.groupby(by):
            # # update masks
            old_names = group[level_name_ini].values.tolist()
            # for mask_name in self.masks.get_mask_names(sample_label=old_names):
            #     mask = self.masks.get_mask(mask_name=mask_name, sample_label=old_names)
            #     self.masks.remove_masks(mask_name=mask_name, sample_label=old_names)
            #     self.masks.add_mask(Mask(mask_name, new_name, mask))
            # merge signal values and betas
            if len(old_names) == 1:
                signal_df_list[new_name] = signal_df[old_names[0]]
                if beta_df is not None:
                    beta_df_list[new_name] = beta_df[old_names[0]]
            else:
                signal_df_list[new_name] = signal_df[old_names].T.groupby(['signal_channel', 'methylation_state']).mean().T
                if beta_df is not None:
                    beta_df_list[new_name] = beta_df[old_names].mean(axis=1)

        # concatenate the results and udpate the attributes
        new_signal_df = pd.concat(signal_df_list, axis=1)
        new_signal_df['mask_info'] = signal_df['mask_info']
        new_signal_df.columns = new_signal_df.columns.rename(by, level=0)
        self._signal_df = new_signal_df
        # same with beta values
        if beta_df is not None:
            new_beta_df = pd.concat(beta_df_list, axis=1)
            new_beta_df.columns = beta_df_list.keys()
            self._betas = new_beta_df

        # and merge the sample sheet
        self.sample_sheet = merge_dataframe_by(self.sample_sheet, by).reset_index()

    def remove_probes_suffix(self, apply_mask=True):
        """Merge probes that have the same ID but different suffixes (e.g. _BC11, _TC21..) by averaging their signal
        values. Resets calculated pvalues and betas.

        :param apply_mask: skip masked probes values when merging samples if True. Default: True
        :type apply_mask: bool
        """
        # todo: optimize this function
        LOGGER.info('Merge probes suffixes..')
        index_cols = ['type', 'channel', 'probe_type', 'probe_id']
        self.reset_poobah()
        self.reset_betas()

        sigdf = self.get_signal_df(apply_mask=apply_mask).reset_index().sort_index(axis=1)
        sigdf.probe_id = sigdf.probe_id.map(remove_probe_suffix)
        dup_indexes = sigdf.probe_id.duplicated(keep=False)
        LOGGER.info('Average duplicated probes values..')
        # todo instead of using merge_dataframe_by, use the values of the probe with the best poobah pvalue (as ChAMP)
        sigdf_merged = merge_dataframe_by(sigdf.loc[dup_indexes], index_cols, observed=True, sort=False, dropna=False, as_index=False)
        self._signal_df = pd.concat([sigdf.loc[~dup_indexes], sigdf_merged], ignore_index=True)
        self._signal_df = self._signal_df.set_index(index_cols).sort_index()

        LOGGER.info('Update masks with merged probes')
        for mask in self.masks:
            masked_probes = mask.indexes.to_frame().reset_index(drop=True)
            masked_probes.probe_id = masked_probes.probe_id.map(remove_probe_suffix)
            mask.indexes = pd.MultiIndex.from_frame(masked_probes).drop_duplicates()

        LOGGER.info('Update annotations')
        self.annotation.probe_infos.probe_id = self.annotation.probe_infos.probe_id.map(remove_probe_suffix)
        self.annotation.genomic_ranges.index = self.annotation.genomic_ranges.index.map(remove_probe_suffix)
        self.annotation.genomic_ranges = self.annotation.genomic_ranges.reset_index().drop_duplicates(ignore_index=True).set_index('probe_id')

    def drop_samples(self, sample_labels: str | list[str]) -> None:
        """Remove some samples. Delete the signal information, beta values, sample sheet rows and masks. Ignores
        non-existent sample names

        :param sample_labels: list of the labels of the samples to drop
        :type sample_labels: str | list[str]

        :return: None
        """
        if isinstance(sample_labels, str):
            sample_labels = [sample_labels]

        self._signal_df = self._signal_df.drop(columns=sample_labels, errors='ignore')
        if self._betas is not None:
            self._betas = self._betas.drop(columns=sample_labels, errors='ignore')
        self.masks.remove_masks(sample_label=sample_labels)
        self.sample_sheet = self.sample_sheet[~self.sample_sheet[self.sample_label_name].isin(sample_labels)].copy()
        # remove unused categories from sample sheet
        for c in self.sample_sheet.select_dtypes('category').columns:
            self.sample_sheet[c] = self.sample_sheet[c].cat.remove_unused_categories()

    def subset(self, sample_labels: str | list[str]) -> None:
        """Keep only the specified samples. Delete the signal information, beta values, sample sheet rows and masks of
        all the samples that are not in the list. Ignores non-existent sample names

        :param sample_labels: list of the labels of the samples to keep
        :type sample_labels: str | list[str]

        :return: None
        """
        if isinstance(sample_labels, str):
            sample_labels = [sample_labels]

        to_drop = [sl for sl in self.sample_labels if sl not in sample_labels]
        self.drop_samples(to_drop)

    ####################################################################################################################
    # Mask functions
    ####################################################################################################################

    def mask_probes_by_names(self, names_to_mask: str | list[str], sample_label: str | None = None, mask_name: str | None = None) -> None:
        """Match the names provided in `names_to_mask` with the probes mask info and mask these probes, adding them to
        the current mask if there is any.

        :param names_to_mask: can be a regex
        :type names_to_mask: str | list[str]
        :param sample_label: The name of the sample to get masked indexes for. If None, returns masked indexes for all samples.
        :type sample_label: str | None
        :param mask_name: the name of the mask to create. If None, the name of the mask will be the same as the names to mask
        :type mask_name: str | None
        :return: None"""

        if isinstance(names_to_mask, list):
            for name in names_to_mask:
                self.mask_probes_by_names(name, sample_label)
            return

        if not isinstance(names_to_mask, str):
            LOGGER.error(f'names_to_mask should be a string or a list of strings, not {type(names_to_mask)}')
            return

        if names_to_mask is None or len(names_to_mask) == 0:
            LOGGER.warning('No names to mask')
            return

        if mask_name is None:
            mask_name = names_to_mask
        new_mask = Mask(mask_name, sample_label, self._signal_df.index[self._signal_df.mask_info.str.contains(names_to_mask)])

        self.masks.add_mask(new_mask)

    def mask_quality_probes(self, sample_label: str | None = None) -> None:
        """Shortcut to mask quality probes

        :param sample_label: The name of the sample to mask. If None, mask indexes for all samples.
        :type sample_label: str | None

        :return: None"""
        self.mask_probes_by_names(self.annotation.quality_mask_names, sample_label, 'quality')

    def mask_non_unique_probes(self, sample_label: str | None = None) -> None:
        """Shortcut to mask non-unique probes on this sample

        :param sample_label: The name of the sample to mask. If None, mask indexes for all samples.
        :type sample_label: str | None
        :return: None"""
        self.mask_probes_by_names(self.annotation.non_unique_mask_names, sample_label, 'non-unique')

    def mask_xy_probes(self, sample_label: str | None = None) -> None:
        """Shortcut to mask probes from XY chromosome

        :param sample_label: The name of the sample to mask. If None, mask indexes for all samples.
        :type sample_label: str | None

        :return: None"""
        xy_probes_ids = self.annotation.probe_infos.probe_id[self.annotation.probe_infos.chromosome.isin(['X', 'Y'])]
        xy_mask = self._signal_df.index[self._signal_df.index.get_level_values('probe_id').isin(xy_probes_ids)]
        self.masks.add_mask(Mask('XY', sample_label, xy_mask))

    def mask_control_probes(self, sample_label: str | None = None) -> None:
        """Shortcut to mask control probes

        :param sample_label: The name of the sample to mask. If None, mask indexes for all samples.
        :type sample_label: str | None

        :return: None"""
        control_probe_ids = self.annotation.probe_infos.probe_id[self.annotation.probe_infos.probe_type == 'ctl']
        control_mask = self._signal_df.index[self._signal_df.index.get_level_values('probe_id').isin(control_probe_ids)]
        self.masks.add_mask(Mask('Control', sample_label, control_mask))

    def mask_snp_probes(self, sample_label: str | None = None) -> None:
        """Shortcut to mask snp probes

        :param sample_label: The name of the sample to mask. If None, mask indexes for all samples.
        :type sample_label: str | None

        :return: None"""
        snp_probe_ids = self.annotation.probe_infos.probe_id[self.annotation.probe_infos.probe_type == 'snp']
        snp_mask = self._signal_df.index[self._signal_df.index.get_level_values('probe_id').isin(snp_probe_ids)]
        self.masks.add_mask(Mask('SNP', sample_label, snp_mask))

    def mask_non_cg_probes(self, sample_label: str | None = None) -> None:
        """Shortcut to mask non-CpG probes

        :param sample_label: The name of the sample to mask. If None, mask indexes for all samples.
        :type sample_label: str | None

        :return: None"""
        # mask control probes separately as we need to be able to access them sometimes, e.g. for normalization
        self.mask_control_probes(sample_label)
        noncg_probe_ids = self.annotation.probe_infos.probe_id[~self.annotation.probe_infos.probe_type.isin(['cg', 'ctl'])]
        noncg_mask = self._signal_df.index[self._signal_df.index.get_level_values('probe_id').isin(noncg_probe_ids)]
        self.masks.add_mask(Mask('NonCG', sample_label, noncg_mask))

    ####################################################################################################################
    # Control functions
    ####################################################################################################################

    def controls(self, apply_mask: bool = True, pattern: str | None = None, sigdf:pd.DataFrame| None=None) -> pd.DataFrame | None:
        """Get the subset of control probes, matching the pattern with the probe_ids if a pattern is provided

        :param apply_mask: True removes masked probes, False keeps them. Default: True
        :type apply_mask: bool

        :param pattern: pattern to match against control probe IDs, case is ignored. Default: None
        :type pattern: str | None

        :param sigdf: signal dataframe to use. Useful to save time applying the mask. Default: None
        :type sigdf: pd.DataFrame | None

        :return: methylation signal dataframe of the control probes, or None if None was found
        :rtype: pandas.DataFrame | None
        """
        if apply_mask:
            initial_mask = self.masks.copy()
            self.masks.remove_masks(mask_name='Control')
            control_df = self.get_probes_with_probe_type('ctl', True, sigdf=sigdf)
            self.masks = initial_mask
        else:
            control_df = self.get_probes_with_probe_type('ctl', False, sigdf=sigdf)

        if control_df is None or len(control_df) == 0:
            LOGGER.info('No control probes found')
            return None

        if pattern is None:
            return control_df  # [['R', 'G']]

        probe_ids = control_df.index.get_level_values('probe_id')
        matched_ids = probe_ids.str.contains(pattern, flags=re.IGNORECASE)
        return control_df[matched_ids]  # [['R', 'G']]

    def get_normalization_controls(self, apply_mask: bool = True, average=False, sigdf:pd.DataFrame| None=None) -> dict | pd.DataFrame | None:
        """Returns the control values to normalize green and red probes.

        :param apply_mask: True removes masked probes, False keeps them. Default: True
        :type apply_mask: bool

        :param average: if set to True, returns a dict with keys 'G' and 'R' containing the average of the control
            probes. Otherwise, returns a dataframe with selected probes. Default: False
        :type average: bool

        :param sigdf: signal dataframe to use. Useful to save time applying the mask. Default: None
        :type sigdf: pd.DataFrame | None

        :return: the normalization controls as a dict or a dataframe, or None if None were found
        :rtype: dict | pandas.DataFrame | None
        """
        if sigdf is None:
            sigdf = self.get_signal_df(apply_mask)

        if self.controls(sigdf=sigdf) is None:
            return None

        # patterns to find the probe IDs we need
        if self.annotation == ArrayType.HUMAN_27K:
            pattern_green = r'norm.green$'
            pattern_red = r'norm.red$'
        else:
            pattern_green = r'norm_c|norm_g$'
            pattern_red = r'norm_a|norm_t$'

        # find the red and green norm control probes according to their probe ID, and set the channel accordingly
        norm_green_df = self.controls(pattern=pattern_green, sigdf=sigdf)
        norm_red_df = self.controls(pattern=pattern_red, sigdf=sigdf)

        if len(norm_green_df) == 0 or len(norm_red_df) == 0:
            LOGGER.warning('No normalization control probes found for at least one channel')
            return None

        if average:
            return { 'G': norm_green_df.xs(('G', 'M'), level=('signal_channel', 'methylation_state'), axis=1).mean(),
                     'R': norm_red_df.xs(('R', 'U'), level=('signal_channel', 'methylation_state'), axis=1).mean()}

        # set channel values to 'G' and 'R' for the green and red control probes respectively
        norm_green_df = norm_green_df.rename(index={np.nan: 'G'}, level='channel')
        norm_red_df = norm_red_df.rename(index={np.nan: 'R'}, level='channel')

        norm_controls = pd.concat([norm_green_df, norm_red_df])

        return norm_controls

    def get_negative_controls(self, apply_mask: bool = True, sigdf:pd.DataFrame| None=None) -> pd.DataFrame | None:
        """Get negative control signal

        :param apply_mask: True removes masked probes, False keeps them. Default: True
        :type apply_mask: bool

        :param sigdf: signal dataframe to use. Useful to save time applying the mask. Default: None
        :type sigdf: pd.DataFrame | None
        
        :return: the negative controls, or None if None were found
        :rtype: pandas.DataFrame | None
        """
        return self.controls(apply_mask, 'negative', sigdf)

    ####################################################################################################################
    # Channel functions
    ####################################################################################################################


    def infer_type1_channel(self, sample_labels: str | list[str] | None = None, switch_failed=False, mask_failed=False, summary_only=False) -> pd.DataFrame:
        """For Infinium type I probes, infer the channel from the signal values, setting it to the channel with the max
        signal. If max values are equals, the channel is set to R (as opposed to G in sesame).

        :param sample_labels: the name(s) of the sample(s) to infer the channel for. If None, infer with all samples. Default: None
        :type sample_labels: str | list[str] | None
        :param switch_failed: if set to True, probes with NA values or whose max values are under a threshold (the 95th
            percentile of the background signals) will be switched back to their original value. Default: False.
        :type switch_failed: bool
        :param mask_failed: mask failed probes (same probes as switch_failed). Default: False.
        :type mask_failed: bool
        :param summary_only: does not replace the sample dataframe, only return the summary (useful for QC). Default:
            False
        :type summary_only: bool

        :return: the summary of the switched channels
        :rtype: pandas.DataFrame"""

        LOGGER.info('Infer type I channel..')
        # reset betas as we are modifying the signal dataframe
        self.reset_betas()

        if sample_labels is None:
            sample_labels = self.sample_labels
        elif isinstance(sample_labels, str):
            sample_labels = [sample_labels]

        # subset to use
        non_na_indexes = self._signal_df[~self._signal_df[sample_labels].isnull().all(axis=1)].index
        non_na_t1_indexes = non_na_indexes[non_na_indexes.get_level_values('type') == 'I']
        type1_df = self._signal_df.loc[non_na_t1_indexes, sample_labels].droplevel('methylation_state', axis=1)

        # get the channel (provided by the index) where the signal is at its max for each probe
        type1_df['inferred_channel'] = type1_df.droplevel(0, axis=1).idxmax(axis=1, numeric_only=True).values
        type1_df = type1_df.sort_index(axis=1)

        # handle failed probes
        if not switch_failed or mask_failed:
            # calculate background for type I probes
            bg_signal_values = np.concatenate([type1_df.loc[type1_df.inferred_channel == 'R', (slice(None), 'G')],
                                               type1_df.loc[type1_df.inferred_channel == 'G', (slice(None), 'R')]])

            bg_max = np.nanpercentile(bg_signal_values, 95)
            failed_idxs = (type1_df.max(axis=1, numeric_only=True) < bg_max) | (type1_df.isna().any(axis=1))

            # reset color channel to the value of 'manifest_channel' for failed indexes of type I probes
            if not switch_failed:
                type1_df.loc[failed_idxs, 'inferred_channel'] = type1_df[failed_idxs].index.get_level_values('channel')

            # mask failed probes
            if mask_failed:
                # failed_ids misses the "type" index level, so we need to add it back - maybe there is a better way
                # failed_probe_ids = failed_idxs[failed_idxs].index.get_level_values('probe_id')
                failed_probe_ids = failed_idxs.index.get_level_values('probe_id')
                mask_indexes = self._signal_df.index[self._signal_df.index.get_level_values('probe_id').isin(failed_probe_ids)]
                self.masks.add_mask(Mask('failed_probes_inferTypeI', None, mask_indexes))

        # set the inferred channel as the new 'channel' index
        if not summary_only:
            # propagate the inferred channel to the signal dataframe
            self._signal_df.loc[non_na_t1_indexes, 'inferred_channel'] = type1_df['inferred_channel'].values
            self._signal_df = set_channel_index_as(self._signal_df, 'inferred_channel', drop=True)  # make the inferred channel the new channel index
            self._signal_df = self._signal_df.sort_index(axis=1)

            # propagate the inferred channel to the masks indexes
            probe_ids = self._signal_df.index.get_level_values('probe_id')
            for mask in self.masks:
                masked_probes = mask.indexes.get_level_values('probe_id')
                mask.indexes = self._signal_df.index[probe_ids.isin(masked_probes)]

        cols = ['channel', 'inferred_channel']
        return type1_df['inferred_channel'].reset_index().groupby(cols, observed=True).count()['probe_id']

    ####################################################################################################################
    # Preprocessing functions
    ####################################################################################################################

    def get_mean_ib_intensity(self, sample_label: str | None = None, apply_mask=True) -> dict:
        """Computes the mean intensity of all the in-band measurements. This includes all Type-I in-band measurements
        and all Type-II probe measurements. Both methylated and unmethylated alleles are considered.

        :param sample_label: the name of the sample to get mean in-band intensity values for. If None, return mean
            in-band intensity values for every sample.
        :type sample_label: str | None
        :param apply_mask: set to False if you don't want any mask to be applied. Default: True
        :type apply_mask: bool

        :return: mean in-band intensity value
        :rtype: float """

        sample_labels = [sample_label] if isinstance(sample_label, str) else self.sample_labels

        sig_df = self.get_signal_df(apply_mask)[sample_labels]  # get signal dataframe for the sample(s)

        # set out-of-band signal to None
        sig_df.loc[(slice(None), 'G'), (slice(None), 'R')] = None
        sig_df.loc[(slice(None), 'R'), (slice(None), 'G')] = None

        mean_intensity = dict()
        for sample_label in sample_labels:
            mean_intensity[sample_label] = sig_df[sample_label][['R', 'G']].mean(axis=None, skipna=True) #  masked rows stay NaN

        return mean_intensity

    def get_total_ib_intensity(self, sample_label: str | list[str] | None = None, apply_mask: bool = True) -> pd.DataFrame:
        """Computes the total intensity of all the in-band measurements. This includes all Type-I in-band measurements
        and all Type-II probe measurements. Both methylated and unmethylated alleles are considered.

        :param sample_label: the name of the sample to get total in-band intensity values for. If None, return total
            in-band intensity values for every sample.
        :type sample_label: str | None

        :param apply_mask: set to False if you don't want any mask to be applied. Default: True
        :type apply_mask: bool

        :return: the total in-band intensity values
        :rtype: pandas.DataFrame"""

        if isinstance(sample_label, str):
            sample_label = [sample_label]
        elif sample_label is None:
            sample_label = self.sample_labels

        sigdf = self.get_signal_df(apply_mask)[sample_label]

        ib_red = self.ib_red(sigdf=sigdf)
        ib_green = self.ib_green(sigdf=sigdf)
        type2 = self.type2(sigdf=sigdf)

        ib = pd.concat([ib_red, ib_green, type2])
        tot_ib_intensity = ib.T.groupby(self.sample_label_name).sum().T
        tot_ib_intensity[tot_ib_intensity == 0] = None
        return tot_ib_intensity

    def calculate_betas(self, include_out_of_band=False) -> None:
        """Calculate beta values for all probes. Values are stored in a dataframe and can be accessed via the betas()
        function

        :param include_out_of_band: is set to true, the Type 1 probes beta values will be calculated on
            in-band AND out-of-band signal values. If set to false, they will be calculated on in-band values only.
            equivalent to sumTypeI in sesame. Default: False
        :type include_out_of_band: bool

        :return: None"""
        LOGGER.info('Calculate beta values')
        df = self.get_signal_df(False).sort_index()  # sort indexes returns a copy
        
        idx = pd.IndexSlice
        # set NAs for Type II probes to 0, only where no methylation signal is expected
        df.loc['II', idx[:, 'R', 'M']] = 0
        df.loc['II', idx[:, 'G', 'U']] = 0

        # set out-of-band signal to 0 if the option include_out_of_band is not activated
        if not include_out_of_band:
            df.loc[idx['I', 'G'], idx[:, 'R']] = 0
            df.loc[idx['I', 'R'], idx[:, 'G']] = 0

        betas = []
        for sample_label in self.sample_labels:
            # now we can calculate beta values
            methylated_signal = df[sample_label, 'R', 'M'] + df[sample_label, 'G', 'M']
            unmethylated_signal = df[sample_label, 'R', 'U'] + df[sample_label, 'G', 'U']

            # use clip function to set minimum values for each term as set in sesame
            betas.append(methylated_signal.clip(lower=1) / (methylated_signal + unmethylated_signal).clip(lower=2))

        self._betas = pd.concat(betas, axis=1)
        self._betas.columns = self.sample_labels
        self._betas = self._betas.sort_index(axis=1)

    def reset_poobah(self) -> None:
        """Remove poobah p-values from the signal dataframe

        :return: None"""
        self._signal_df = self._signal_df.drop(['p_value'], level='signal_channel', axis=1, errors='ignore')

    def reset_betas(self) -> None:
        """Remove betas dataframe

        :return: None"""
        self._betas = None

    def has_betas(self) -> bool:
        """Return True if the beta values have already been calculated

        :return: bool"""

        return self._betas is not None

    def get_betas(self, sample_label: str | None = None, drop_na: bool = False, probe_ids: list[str] | str | None=None,
                  custom_sheet: pd.DataFrame | None = None, apply_mask: bool=True) -> pd.DataFrame | pd.Series | None:
        """Get the beta values for the sample. If no sample name is provided, return beta values for all samples.

        :param sample_label: the name of the sample to get beta values for. If None, return beta values for all samples.
        :type sample_label: str | None
        :param drop_na: if set to True, drop rows with NA values. Default: False
        :type drop_na: bool
        :param custom_sheet: a custom sample sheet to filter samples. Ignored if sample_label is provided. Default: None
        :type custom_sheet: pandas.DataFrame | None
        :param apply_mask: set to False if you don't want any mask to be applied. Default: True
        :type apply_mask: bool
        :param probe_ids: the IDs of the probes to select
        :type probe_ids: list[str]
    
        :return: beta values as a DataFrame, or Series if sample_label is provided. If no beta values are found, return None
        :rtype: pandas.DataFrame | pandas.Series | None"""

        if not self.has_betas():
            LOGGER.error('No beta values found. Please calculate beta values first')
            return None

        betas = self._betas.copy()

        if apply_mask:
            beta_idx = betas.index
            for label in self.sample_labels:
                sample_mask = self.masks.get_mask(sample_label=label)
                if sample_mask is not None and len(sample_mask) > 0:
                    sample_mask = sample_mask[sample_mask.isin(beta_idx)]
                    if len(sample_mask) > 0:
                        betas.loc[sample_mask, label] = None

        if custom_sheet is not None and sample_label is not None:
            LOGGER.warning('Both sample_label and custom_sheet are provided. Ignoring custom_sheet')
            custom_sheet = None

        if custom_sheet is not None:
            if len(custom_sheet) == 0:
                LOGGER.error('Empty custom_sheet')
                return None

            if self.sample_label_name not in custom_sheet.columns:
                LOGGER.error(f'No {self.sample_label_name} column found in custom_sheet ({custom_sheet.columns})')
                return None

            # keep only samples that are both in sample sheet and beta columns
            filtered_samples = [col for col in custom_sheet[self.sample_label_name].values if col in betas.columns]
            if len(filtered_samples) == 0:
                LOGGER.error('No samples found')
                return None
            betas = betas[filtered_samples]

        if sample_label is not None:
            if sample_label not in betas.columns:
                LOGGER.error(f'Sample {sample_label} not found')
                return None
            betas = betas[sample_label]

        if drop_na:
            betas = betas.dropna()

        if probe_ids is not None and len(probe_ids) > 0:

            if isinstance(probe_ids, str):
                probe_ids = [probe_ids]

            return betas[betas.index.get_level_values('probe_id').isin(probe_ids)]

        return betas

    @staticmethod
    def _betas_to_m(beta_df: pd.DataFrame) -> pd.DataFrame:
        """Convert the beta values dataframe to M-values
        :param beta_df: beta values
        :type beta_df: pd.DataFrame
        
        :return: M-values
        :rtype: pd.DataFrame
        """
        epsilon = 1e-8  # add a small epsilon to avoid log2(0) or division by zero
        x = beta_df.values + epsilon  # convert to NumPy array for speed
        m_values = np.log2((x) / (1 - x))
        return pd.DataFrame(m_values, index=beta_df.index, columns=beta_df.columns)

    @staticmethod
    def _m_to_betas(m_df: pd.DataFrame) -> pd.DataFrame:
        """Convert M-values (logit-transformed beta values) back to beta values.
        :param m_df: M-values
        :type m_df: pd.DataFrame
        
        :return: beta values in [0, 1]
        :rtype: pd.DataFrame
        """
        m = m_df.values if isinstance(m_df, pd.DataFrame) else m_df
        beta = 1 / (1 + 2**(-m))
        if isinstance(m_df, pd.DataFrame):
            return pd.DataFrame(beta, index=m_df.index, columns=m_df.columns)
        return beta
    
    def get_m_values(self, sample_label: str | None = None, drop_na: bool = False, probe_ids: list[str] | str | None=None,
                  custom_sheet: pd.DataFrame | None = None, apply_mask: bool=True) -> pd.DataFrame | pd.Series | None:
        """Get the M-values for the sample. If no sample name is provided, return M-values for all samples. They are 
        calculated from the beta values, so they need to be calculated first using the calculate_betas() function.

        :param sample_label: the name of the sample to get beta values for. If None, return beta values for all samples.
        :type sample_label: str | None
        :param drop_na: if set to True, drop rows with NA values. Default: False
        :type drop_na: bool
        :param custom_sheet: a custom sample sheet to filter samples. Ignored if sample_label is provided. Default: None
        :type custom_sheet: pandas.DataFrame | None
        :param apply_mask: set to False if you don't want any mask to be applied. Default: False
        :type apply_mask: bool
        :param probe_ids: the IDs of the probes to select
        :type probe_ids: list[str]
    
        :return: beta values as a DataFrame, or Series if sample_label is provided. If no beta values are found, return None
        :rtype: pandas.DataFrame | pandas.Series | None"""

        betas_df = self.get_betas(sample_label=sample_label, drop_na=drop_na, probe_ids=probe_ids, custom_sheet=custom_sheet,
                                  apply_mask=apply_mask)
        
        if betas_df is None:
            return None
        

        return self._betas_to_m(betas_df)

    def dye_bias_correction(self, sample_label: str | None = None, apply_mask: bool = True, reference: dict | None = None) -> None:
        """Dye bias correction using normalization control probes.

        :param sample_label: the name of the sample to correct dye bias for. If None, correct dye bias for all samples.
        :type sample_label: str | None

        :param apply_mask: set to False if you don't want any mask to be applied. Default: True
        :type apply_mask: bool

        :param reference: values to use as reference to scale red and green signal for each sample (=dict keys). Default: None
        :type: dict | None

        :return: None
        """
        LOGGER.info('Dye bias correction..')
        self.reset_betas()  # reset betas as we are modifying the signal dataframe
        if sample_label is None:
            for sample_label in self.sample_labels:
                self.dye_bias_correction(sample_label, apply_mask, reference)
            return None

        if not isinstance(sample_label, str):
            LOGGER.error('sample_label should be a string')
            return None

        if sample_label not in self.sample_labels:
            LOGGER.error(f'Sample {sample_label} not found')
            return None

        if reference is None:
            reference = self.get_mean_ib_intensity(sample_label, apply_mask)

        norm_values_dict = self.get_normalization_controls(apply_mask, average=True)

        if norm_values_dict is None:
            return None

        for channel in ['R', 'G']:
            factor = reference[sample_label] / norm_values_dict[channel][sample_label]
            self._signal_df[(sample_label, channel)] *= factor

        return None

    def dye_bias_correction_l(self, sample_label: str | None = None, apply_mask: bool = True, reference: dict | None = None) -> None:
        """Linear dye bias correction. Scale both the green and red signal to a reference level. If the reference level
         is not given, it is set to the mean intensity of all the in-band signals.

        :param sample_label: the name of the sample to correct dye bias for. If None, correct dye bias for all samples.
        :type sample_label: str | None

        :param apply_mask: set to False if you don't want any mask to be applied. Default: True
        :type apply_mask: bool

        :param reference: values to use as reference to scale red and green signal for each sample (=dict keys). Default: None
        :type: dict | None

        :return: None
        """
        LOGGER.info('Dye bias correction by linear scaling..')
        self.reset_betas()  # reset betas as we are modifying the signal dataframe
        if sample_label is None:
            for sample_label in self.sample_labels:
                self.dye_bias_correction_l(sample_label, apply_mask, reference)
            return

        if not isinstance(sample_label, str):
            LOGGER.error('sample_label should be a string')
            return

        if sample_label not in self.sample_labels:
            LOGGER.error(f'Sample {sample_label} not found')
            return

        if reference is None:
            reference = self.get_mean_ib_intensity(sample_label, apply_mask)

        norm_values_dict = {'R': self.type1_red(apply_mask)[(sample_label, 'R')].median(axis=None),
                            'G': self.type1_green(apply_mask)[(sample_label, 'G')].median(axis=None)}

        for channel in ['R', 'G']:
            factor = reference[sample_label] / norm_values_dict[channel]
            self._signal_df[(sample_label, channel)] *= factor

    def dye_bias_correction_nl(self, sample_labels: str | list[str] | None = None, apply_mask: bool = True) -> None:
        """Non-linear dye bias correction by matching green and red to mid-point. Each sample is handled separately.

        This function compares the Type-I Red probes and Type-I Grn probes and generates and mapping to correct signal
        of the two channels to the middle.

        :param sample_labels: the name of the sample to correct dye bias for. If None, correct dye bias for all samples.
        :type sample_labels: str | list[str] | None

        :param apply_mask: if True include masked probes in Infinium-I probes. No big difference is noted in practice. More
            probes are generally better. Default: True
        :type apply_mask: bool

        :return: None
        """
        LOGGER.info('Non linear dye bias correction..')
        self.reset_betas()  # reset betas as we are modifying the signal dataframe

        if sample_labels is None:
            sample_labels = self.sample_labels
        elif isinstance(sample_labels, str):
            sample_labels = [sample_labels]

        type_1_green = self.type1_green(apply_mask)
        type_1_red = self.type1_red(apply_mask)

        ib_t1_intensity = self.get_total_ib_intensity(apply_mask=False).loc['I'].sort_index()

        # check that there is not too much distortion between the two channels
        for sample_label in sample_labels:
            total_intensity_type1 = ib_t1_intensity[sample_label]

            median_red = np.median(total_intensity_type1.loc['R'])
            median_green = np.median(total_intensity_type1.loc['G'])

            top_20_median_red = np.median(total_intensity_type1.loc['R'].nlargest(20))  # 0.25 sec
            top_20_median_green = np.median(total_intensity_type1.loc['G'].nlargest(20))  # 0.25 sec

            if top_20_median_green == 0 or median_green == 0:
                red_green_distortion = None
            else:
                red_green_distortion = (top_20_median_red / top_20_median_green) / (median_red / median_green)

            if red_green_distortion is None or red_green_distortion is np.nan or red_green_distortion > 10:
                LOGGER.warning(f'Red-Green distortion is too high or None ({red_green_distortion}). Masking green probes')
                type1_mask = self._signal_df.index[self._signal_df.index.get_level_values('channel') == 'G']
                self.masks.add_mask(Mask('dye bias nl', sample_label, type1_mask))
                return

            # all good, we can apply dye bias correction...
            sorted_intensities = {'G': np.sort(get_column_as_flat_array(type_1_green[sample_label], 'G', remove_na=True)),
                                  'R': np.sort(get_column_as_flat_array(type_1_red[sample_label], 'R', remove_na=True))}

            # ... if red or green channel intensities are not all 0
            if np.max(sorted_intensities['G']) <= 0 or np.max(sorted_intensities['R']) <= 0:
                LOGGER.warning('Max green or red intensities is <= 0. Aborting dye bias correction.')
                return

            for channel, reference_channel in [('R', 'G'), ('G', 'R')]:
                channel_intensities = sorted_intensities[channel]
                ref_intensities = sorted_intensities[reference_channel]

                max_intensity = np.max(channel_intensities)
                min_intensity = np.min(channel_intensities)
                normalized_intensities = np.sort(quantile_normalization_using_target(channel_intensities, ref_intensities))
                midpoint_intensities = (channel_intensities + normalized_intensities) / 2
                max_midpoint_intensity = np.max(midpoint_intensities)
                min_midpoint_intensity = np.min(midpoint_intensities)

                # check that none of max or min intensities are none
                if None in [max_intensity, min_intensity, max_midpoint_intensity, min_midpoint_intensity]:
                    LOGGER.warning(f'Max or min intensities are None. Aborting dye bias correction for sample {sample_label}.')
                    continue

                def fit_function(data: np.array) -> np.array:
                    within_range = (data <= max_intensity) & (data >= min_intensity) & (~np.isnan(data))
                    above_range = (data > max_intensity) & (~np.isnan(data))
                    below_range = (data < min_intensity) & (~np.isnan(data))
                    data[within_range] = np.interp(x=data[within_range], xp=channel_intensities, fp=midpoint_intensities)
                    data[above_range] = data[above_range] - max_intensity + max_midpoint_intensity
                    if min_intensity == 0:
                        data[below_range] = np.nan
                    else:
                        data[below_range] = data[below_range] * (min_midpoint_intensity / min_intensity)
                    return data

                self._signal_df.loc[:, [(sample_label, channel, 'M')]] = fit_function(self._signal_df[[(sample_label, channel, 'M')]].values)
                self._signal_df.loc[:, [(sample_label, channel, 'U')]] = fit_function(self._signal_df[[(sample_label, channel, 'U')]].values)

    def noob_background_correction(self, sample_labels: str | list[str] | None = None, apply_mask: bool = True, use_negative_controls=True, offset=15) -> None:
        """Subtract the background for a sample.

        Background was modelled in a normal distribution and true signal in an exponential distribution. The Norm-Exp
        deconvolution is parameterized using Out-Of-Band (oob) probes. Multi-mapping probes are excluded.

        :param sample_labels: the name(s) of the sample(s) to correct dye bias for. If None, correct dye bias for all samples.
        :type sample_labels: str | list[str] | None

        :param apply_mask: True removes masked probes, False keeps them. Default: True
        :type apply_mask: bool

        :param use_negative_controls: if True, the background will be calculated with both negative control and
            out-of-band probes. Default: True
        :type use_negative_controls: bool

        :param offset: A constant value to add to the corrected signal for padding. Default: 15
        :type offset: int | float

        :return: None
        """
        LOGGER.info('NOOB background correction..')

        self.reset_betas()  # reset betas as we are modifying the signal dataframe

        if sample_labels is None:
            sample_labels = self.sample_labels
        elif isinstance(sample_labels, str):
            sample_labels = [sample_labels]

        # mask non-unique probes - saves previous mask to reset it afterwards
        initial_masks = self.masks.copy()
        if not apply_mask:
            self.masks.reset_masks()
        self.mask_probes_by_names(self.annotation.non_unique_mask_names)

        # Background = out-of-band type 1 probes + (optionally) negative controls
        background_df = self.oob(True)
        if use_negative_controls:
            neg_controls = self.get_negative_controls(True)
            background_df = pd.concat([background_df, neg_controls])

        # Foreground = in-band type I probes + type 2 probes
        foreground_df = pd.concat([self.ib(True), self.type2(True)])

        # reset apply_mask
        self.masks = initial_masks

        for sample_label in sample_labels:

            bg = dict()
            fg = dict()

            for channel in ['R', 'G']:
                bg[channel] = get_column_as_flat_array(background_df[sample_label], channel, remove_na=True)
                fg[channel] = get_column_as_flat_array(foreground_df[sample_label], channel, remove_na=True)

                if len(bg[channel][bg[channel] > 0]) < 100:
                    LOGGER.warning('Not enough out of band signal to perform NOOB background subtraction')
                    return

                bg[channel][bg[channel] == 0] = 1
                fg[channel][fg[channel] == 0] = 1

                # cap at 10xIQR, this is to proof against multi-mapping probes
                bg[channel] = bg[channel][bg[channel] < (np.median(bg[channel]) + 10 * iqr(bg[channel]))]

                mu, sigma, alpha = background_correction_noob_fit(fg[channel], bg[channel])
                sample_df = self._signal_df[sample_label, channel]
                meth_corrected_signal = norm_exp_convolution(mu, sigma, alpha, sample_df['M'].values, offset)
                unmeth_corrected_signal = norm_exp_convolution(mu, sigma, alpha, sample_df['U'].values, offset)

                self._signal_df.loc[:, [[sample_label, channel, 'M']]] = meth_corrected_signal
                self._signal_df.loc[:, [[sample_label, channel, 'U']]] = unmeth_corrected_signal

    def scrub_background_correction(self, sample_label: str | None = None,  apply_mask: bool = True) -> None:
        """Subtract residual background using background median.

        This function is meant to be used after noob.

        :param  sample_label: the name of the sample to scrub background for. If None, scrub background for all samples.
        :type sample_label: str | None
        :param apply_mask: True removes masked probes, False keeps them. Default: True
        :type apply_mask: bool

        :return: None"""
        LOGGER.info('Scrub background correction..')
        self.reset_betas()  # reset betas as we are modifying the signal dataframe
        sample_labels = [sample_label] if isinstance(sample_label, str) else self.sample_labels

        for sample_label in sample_labels:

            median_bg = {'G': self.oob_green(apply_mask)[sample_label].median(axis=None),
                         'R': self.oob_red(apply_mask)[sample_label].median(axis=None)}

            for channel in ['G', 'R']:
                for methylation_state in ['U', 'M']:
                    idx = [(sample_label, channel, methylation_state)]
                    self._signal_df.loc[:, idx] = np.clip(self._signal_df[idx] - median_bg[channel], a_min=1, a_max=None)

    def poobah(self, sample_labels: str | list[str] | None = None, apply_mask: bool = True, use_negative_controls=True, threshold=0.05) -> None:
        """Detection P-value based on empirical cumulative distribution function (ECDF) of out-of-band signal
        aka pOOBAH (P-value with out-of-band (OOB) array hybridization). Each sample is handled separately.

        Adds two columns in the signal dataframe, 'p_value' and 'poobah_mask'. Add probes that are (strictly) above the
        defined threshold to the mask.

        :param sample_labels: the name(s) of the sample(s) to use for the pOOBAH calculation. If None, use all samples.
            Default: None
        :type sample_labels: str | list[str] | None

        :param apply_mask: True removes masked probes from background, False keeps them. Default: True
        :type apply_mask: bool

        :param use_negative_controls: add negative controls as part of the background. Default True
        :type use_negative_controls: bool

        :param threshold: used to output a mask based on the p_values.
        :type threshold: float

        :return: None"""

        LOGGER.info('start pOOBAH')
        self.reset_poobah()

        # mask non-unique probes - but first save previous mask to reset it afterward
        initial_masks = self.masks.copy()

        if not apply_mask:
            self.masks.reset_masks()

        self.mask_non_unique_probes()

        # Background = out-of-band type 1 probes + (optionally) negative controls
        masked_sigdf = self.get_signal_df(True)
        background_df = self.oob(sigdf=masked_sigdf)
        if use_negative_controls:
            neg_controls = self.get_negative_controls(sigdf=masked_sigdf)
            background_df = pd.concat([background_df, neg_controls])

        # reset apply_mask
        self.masks = initial_masks

        if isinstance(sample_labels, str):
            sample_labels = [sample_labels]
        elif sample_labels is None:
            sample_labels = self.sample_labels

        pvalues = []
        new_colnames = []
        for sample_label in sample_labels:
            bg_green = get_column_as_flat_array(background_df[sample_label], 'G', remove_na=True)
            bg_red = get_column_as_flat_array(background_df[sample_label], 'R', remove_na=True)

            if np.sum(bg_red, where=~np.isnan(bg_red)) <= 100:
                LOGGER.debug('Not enough out of band signal, use empirical prior')
                bg_red = [n for n in range(1000)]

            if np.sum(bg_green, where=~np.isnan(bg_green)) <= 100:
                LOGGER.debug('Not enough out of band signal, use empirical prior')
                bg_green = [n for n in range(1000)]

            pval_green = 1 - ecdf(bg_green)(self._signal_df[(sample_label, 'G')].max(axis=1))
            pval_red = 1 - ecdf(bg_red)(self._signal_df[(sample_label, 'R')].max(axis=1))

            # set new columns with pOOBAH values
            pvalues.append(pd.Series(np.min([pval_green, pval_red], axis=0)))
            new_colnames.append((sample_label, 'p_value', ''))

        pval_df = pd.concat(pvalues, axis=1)
        pval_df.columns = pd.MultiIndex.from_tuples(new_colnames, names=self._signal_df.columns.names)
        pval_df.index = self._signal_df.index
        self._signal_df = pd.concat([self._signal_df, pval_df], axis=1).sort_index(axis=1)

        for sample_label in sample_labels:
            # add a mask for the sample, depending on the threshold
            mask_indexes = self._signal_df.index[self._signal_df[(sample_label, 'p_value', '')] >= threshold]
            poobah_mask = Mask(f'poobah_{threshold}', sample_label, mask_indexes)
            self.masks.add_mask(poobah_mask)

    def batch_correction(self, batch:list|str, apply_mask:bool=True, covariates:str|list[str]|None=None,
                         par_prior=True, mean_only=False, ref_batch=None, precision=None, na_cov_action='raise') -> None:
        """Applies ComBat algorithm for batch correction. To correct the beta values while staying in the [0:1] range, 
        the algorithm is applied on M-values, that are converted back to beta values. If the batch correction fails,
        the beta values are reset to None.

        :param batch: If a string is provided, it's interpreted as the name of the column in the sample sheet that
            contains the batch information. If a list is provided, it should contain the batch indices, with as many
            values as samples.
        :type batch: str | list

        :param apply_mask: set to False if you don't want any mask to be applied. Default: True
        :type apply_mask: bool

        :param covariates: a list of column names from the sample sheet to use as covariates in the model. It only
            supports categorical or string variables. Default: None
        :type covariates: str | list[str] | None

        :param par_prior: False for non-parametric estimation of batch effects. Default: True
        :type par_prior: bool

        :param mean_only: True iff just adjusting the means and not individual batch effects. Default: False
        :type mean_only: bool

        :param ref_batch: batch id of the batch to use as reference. Default: None

        :param precision: level of precision for precision computing. Default: None
        :type precision: float

        :param na_cov_action: choose the way to handle missing covariates : `raise` raise an error if missing covariates
            and stop the code, `remove` remove samples with missing covariates and raise a warning, `fill` handle missing
            covariates, by creating a distinct covariate per batch. Default: `raise`

        :return: None
        """
        LOGGER.info('>>> Start ComBat batch correction')

        if not self.has_betas():
            LOGGER.error('No beta values found. Please calculate beta values first')
            return

        sheet = self.sample_sheet.set_index(self.sample_label_name)
        sheet = sheet.loc[self._betas.columns]  # sort like betas

        # get batch from the batch column, if specified
        if isinstance(batch, str):
            if batch not in sheet.columns:
                LOGGER.error(f'Batch column {batch} not found in sample sheet')
                self.reset_betas()
                return
            batch = sheet[batch].values

        if np.any(pd.isnull(batch)) or np.any(batch == ''):
            LOGGER.error('Batch column contains NaN or empty values')
            self.reset_betas()
            return

        if len(batch) != len(self._betas.columns):
            LOGGER.error('Batch column length does not match the number of samples. This can happen if some samples have duplicate names.')
            self.reset_betas()
            return

        # if any covariates are specified, check that they exist in the sample sheet and have the right type
        if covariates is not None:
            if isinstance(covariates, str):
                covariates = [covariates]
            checked_covariates = []
            for cov in covariates:
                if cov not in sheet.columns:
                    LOGGER.warning(f'Covariate {cov} not found in sample sheet. Ignoring it.')
                elif sheet[cov].dtype not in ['object', 'category']:
                    LOGGER.warning(f'Covariate {cov} must be a string or a category. Ignoring it.')
                else:
                    checked_covariates.append(cov)
            if len(checked_covariates) == 0:
                LOGGER.warning('No valid covariates found. Ignoring covariates.')
                covariates = None
            else:
                covariates = sheet[checked_covariates]
                LOGGER.info(f'Using covariates {checked_covariates}')

        m_values = self._betas_to_m(self.get_betas(apply_mask=apply_mask).dropna())
        
        try:
            m_values = pycombat_norm(m_values, batch,
                                    covar_mod=covariates, par_prior=par_prior, mean_only=mean_only, ref_batch=ref_batch,
                                    precision=precision, na_cov_action=na_cov_action)
            self._betas = self._m_to_betas(m_values)

        except np.linalg.LinAlgError as e:
            LOGGER.error(f'Batch correction failed due to {e}. This is most likely due to the experiment design leading to' \
                          'linear dependencies. Please check your data and/or try to run with different or no covariates.')
            self.reset_betas()
        
    def get_nb_probes_per_chr_and_type(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Count the number of probes covered by the sample-s per chromosome and design type

        :return: two dataframes: number of probes per chromosome and number of probes per design type (masked and not masked) """

        chromosome_df = pd.DataFrame(columns=['not masked', 'masked'])
        type_df = pd.DataFrame(columns=['not masked', 'masked'])
        manifest = self.annotation.probe_infos[['probe_id', 'chromosome', 'type']].drop_duplicates()
        manifest['chromosome'] = merge_alt_chromosomes(manifest['chromosome'])

        masked_probe_ids = self.masks.get_mask(sample_label=self.sample_labels).get_level_values('probe_id')
        manifest_masked_probes = manifest.probe_id.isin(masked_probe_ids)

        chrm_and_type = manifest.loc[manifest_masked_probes]
        chromosome_df['masked'] = chrm_and_type.groupby('chromosome', observed=True).count()['probe_id']
        type_df['masked'] = chrm_and_type.groupby('type', observed=False).count()['probe_id']
        chrm_and_type = manifest.loc[~manifest_masked_probes]
        chromosome_df['not masked'] = chrm_and_type.groupby('chromosome', observed=True).count()['probe_id']
        type_df['not masked'] = chrm_and_type.groupby('type', observed=False).count()['probe_id']

        # get the chromosomes numbers to order data frame correctly
        chromosome_df['chr_id'] = get_chromosome_number(chromosome_df.index.tolist())
        chromosome_df = chromosome_df.sort_values('chr_id').drop(columns='chr_id')

        type_df.index = [f'Type {i}' for i in type_df.index]
        type_df.index.name = 'Probe type'
        chromosome_df.index.name = 'Chromosome number'

        return chromosome_df, type_df


def read_idata(sample_sheet_df: pd.DataFrame, datadir: str | Path) -> tuple[dict, str]:
    """
    Reads IDAT files for each sample in the provided sample sheet, organizes the data by sample name and channel,
    and returns a dictionary with the IDAT data and the label column name.

    :param sample_sheet_df: A DataFrame containing sample information, including columns for 'sample_label', 'sample_id',
        'sentrix_id', and 'sentrix_position'. Each row corresponds to a sample in the experiment.
    :type  sample_sheet_df: pandas.DataFrame

    :param datadir: The directory where the IDAT files are located.
    :type datadir: str

    :return: A tuple of a dictionary (samples) and a string (label column name).
        The samples dictionary keys are sample names (from the 'sample_label' column in `sample_sheet_df`), and the
        values are dictionaries mapping channel names (from `Channel`) to their respective IDAT data (as DataFrame
        objects, derived from the `IdatDataset` class).
    :rtype: (dict, str)

    Notes:
        - The function searches for IDAT files by sample ID and channel. If no files are found, it attempts to search
          using the Sentrix ID and position.
        - If multiple files match the search pattern, an error is logged.
        - If no matching files are found, an error is logged and the sample is skipped.

    Example:
        idata, label_column = read_idata(sample_sheet_df, '/path/to/data')
    """
    idata = {}
    label_column = 'sample_id'
    if 'sample_name' in sample_sheet_df.columns and sum(sample_sheet_df['sample_name'].duplicated()) == 0:
        label_column = 'sample_name'
    elif sum(sample_sheet_df['sample_id'].duplicated()) > 0:
        LOGGER.warning('Your sample sheet contains duplicated sample_ids. Please check your data as it may lead to errors')

    LOGGER.info(f'Use colum {label_column} to label samples')

    for _, line in sample_sheet_df.iterrows():

        idata[line[label_column]] = dict()

        # read each channel file
        for channel in Channel:
            pattern = f'*{line.sample_id}*{channel}*.idat*'
            paths = [str(p) for p in get_files_matching(datadir, pattern)]
            if len(paths) == 0:
                if line.sentrix_id != '' and line.sentrix_position != '':
                    pattern = f'*{line.sentrix_id}*{line.sentrix_position}*{channel}*.idat*'
                    paths = [str(p) for p in get_files_matching(datadir, pattern)]
            if len(paths) == 0:
                LOGGER.error(f'no paths found matching {pattern}')
                continue
            if len(paths) > 1:
                LOGGER.error(f'Too many files found matching {pattern} : {paths}')
                continue

            idat_filepath = paths[0]
            LOGGER.debug(f'reading file {idat_filepath}')
            idata[line[label_column]][channel] = IdatDataset(idat_filepath).probes_df

    return idata, label_column


def read_samples(datadir: str | os.PathLike | MultiplexedPath,
                 sample_sheet_df: pd.DataFrame | None = None,
                 sample_sheet_name: str | None = None,
                 annotation: Annotations | None = None,
                 max_samples: int | None = None,
                 min_beads=1,
                 keep_idat=False) -> Samples | None:
    """Search for idat files in the datadir through all sublevels.

    The idat files are supposed to match the information from the sample sheet and follow this naming convention:
    `*[sentrix ID]*[sentrix position]*[channel].idat` where `*` can be any characters.
    `channel` must be spelled `Red` or `Grn`.

    :param datadir:  directory where sesame files are
    :type datadir: str  | os.PathLike | MultiplexedPath

    :param sample_sheet_df: samples information. If not given, will be automatically rendered. Default: None
    :type sample_sheet_df: pandas.DataFrame | None

    :param sample_sheet_name: name of the csv file containing the samples' information. You cannot provide both a sample
        sheet dataframe and name. Default: None
    :type sample_sheet_name: str | None

    :param annotation: probes information. Default None.
    :type annotation: Annotations | None

    :param max_samples: set it to only load N samples to speed up the process (useful for testing purposes). Default: None
    :type max_samples: int | None

    :param min_beads: filter probes that have less than N beads. Default: 1
    :type min_beads: int

    :param keep_idat: if set to True, keep idat data after merging the annotations. Default: False
    :type: bool

    :return: Samples object or None if an error was raised
    :rtype: Samples | None"""
    datadir = convert_to_path(datadir)  # expand user and make it a Path
    LOGGER.info(f'>> start reading sample files from {datadir}')

    if sample_sheet_df is not None and sample_sheet_name is not None:
        LOGGER.error('You can\'t provide both a sample sheet dataframe and name. Please only provide one parameter.')
        return None
    elif sample_sheet_df is None and sample_sheet_name is None:
        if os.path.exists(f'{datadir}/samplesheet.csv'):
            LOGGER.info('Found a samplesheet.csv file in data directory')
            sample_sheet_df = sample_sheet.read_from_file(f'{datadir}/samplesheet.csv')
            if sample_sheet_df is None:
                LOGGER.error('Invalid samplesheet.csv file, can\'t read the samples')
                return None
        else:
            LOGGER.info('No sample sheet provided nor found, creating one')
            sample_sheet_df, _ = sample_sheet.create_from_idats(datadir)
    elif sample_sheet_name is not None:
        sample_sheet_df = sample_sheet.read_from_file(f'{datadir}/{sample_sheet_name}')

    if min_beads < 1:
        LOGGER.warning('min_beads must be >= 1. Setting it to 1.')
        min_beads = 1

    # check that the sample sheet was correctly created / read. If not, abort mission.
    if sample_sheet_df is None:
        return None

    sample_sheet_df = sample_sheet_df.drop_duplicates()  # make sure there is no duplicated rows

    # only load the N first samples
    if max_samples is not None:
        sample_sheet_df = sample_sheet_df.sort_values('sample_id').head(max_samples)

    samples = Samples(sample_sheet_df)
    samples.idata, label_column = read_idata(sample_sheet_df, datadir)

    if samples.idata is None or len(samples.idata) == 0:
        LOGGER.error('no idat files found')
        return None

    samples.add_annotation_info(annotation, label_column, keep_idat, min_beads)

    LOGGER.info(f'reading sample files done, {samples.nb_samples} samples found\n')
    return samples

def from_sesame(datadir: str | os.PathLike | MultiplexedPath, annotation: Annotations, no_suffix=False) -> Samples | None:
    """Reads all .csv files in the directory provided, supposing they are SigDF from SeSAMe saved as csv files.

    :param datadir:  directory where sesame files are, or path to a .csv file
    :type datadir: str | os.PathLike | MultiplexedPath
    :param annotation: Annotations object with genome version and array type corresponding to the data stored
    :type annotation: Annotations
    :param no_suffix: set to True if the probe ids from sesame don't have a suffix (i.e. they look like 'cg00000155' and
        not 'cg00000155_BC21'). Default: False
    :type no_suffix: bool

    :return: a Samples object
    :rtype: Samples | None"""

    LOGGER.info('>> start reading sesame files')

    if isinstance(datadir, list):
        LOGGER.error('You can only provide one datadir or filepath for sesame files')
        return None

    # find all .csv files in the subtree depending on datadir type
    datadir = convert_to_path(datadir)
    if '.csv' in str(datadir):
        file_list = [datadir]
    else:
        file_list = get_files_matching(datadir, '*.csv*')

    if len(file_list) == 0:
        LOGGER.error('no csv files found')
        return None

    samples = Samples()
    samples.annotation = annotation
    sample_labels = [f.stem.split('.csv')[0] for f in file_list]
    samples.sample_sheet = pd.DataFrame({'sample_id': sample_labels, 'sample_label': sample_labels})
    dfs = []

    # prepare manifest for merge
    manifest = annotation.probe_infos.loc[:, ['probe_id', 'type', 'probe_type', 'channel', 'mask_info']]
    # remove probe suffix from manifest if specified
    if no_suffix:
        manifest['probe_id'] = manifest.probe_id.apply(remove_probe_suffix)
    manifest = manifest.set_index('probe_id')

    mandatory_columns = ['probe_id', 'MG', 'MR', 'UG', 'UR', 'mask']

    # load all samples
    for csv_file in file_list:

        name = Path(csv_file).stem

        # read input file
        sig_df = pd.read_csv(csv_file, low_memory=False)
        sig_df = sig_df.rename(columns={'Probe_ID': 'probe_id'})

        # check that the csv file has all the mandatory columns
        missing_col = False
        for col in mandatory_columns:
            if col not in sig_df.columns:
                LOGGER.error(f'no "{col}" column found in {csv_file}, skip the file')
                missing_col = True
        if missing_col:
            continue

        # only keep mandatory columns
        sig_df = sig_df.loc[:, mandatory_columns].set_index('probe_id')

        # merge manifest and apply_mask
        sample_df = sig_df.join(manifest, how='inner')

        # move Green type II probes values to MG column
        sample_df.loc[sample_df['type'] == 'II', 'MG'] = sample_df.loc[sample_df['type'] == 'II', 'UG']
        sample_df.loc[sample_df['type'] == 'II', 'UG'] = np.nan

        # set signal channel for type II probes
        sample_df.loc[(sample_df['type'] == 'II') & (sample_df.MG.isna()), 'channel'] = 'R'
        sample_df.loc[(sample_df['type'] == 'II') & (sample_df.UR.isna()), 'channel'] = 'G'

        # make multi-index for rows and columns
        sample_df = sample_df.reset_index().set_index(['type', 'channel', 'probe_type', 'probe_id'])
        sample_df = sample_df.loc[:, ['UR', 'MR', 'MG', 'UG', 'mask', 'mask_info']]  # order columns
        sample_df.columns = pd.MultiIndex.from_tuples([('R', 'U'), ('R', 'M'), ('G', 'M'), ('G', 'U'), ('mask', ''), ('mask_info', '')])

        # set mask as specified in the input file, then drop the mask column
        samples.masks.add_mask(Mask('sesame', name, sample_df.index[sample_df['mask']]))
        sample_df = sample_df.sort_index(axis=1).drop(columns='mask')
        dfs.append(sample_df)

    if len(dfs) == 0:
        return None

    if len(dfs) != len(sample_labels):
        LOGGER.warning(f'{len(dfs)} dfs != {len(sample_labels)} samples names for {datadir}')
        return None

    sig_df = pd.concat(dfs, axis=1, keys=sample_labels).drop_duplicates()
    sig_df.columns.names = ['sample_id', 'signal_channel', 'methylation_state']
    samples._signal_df = sig_df

    LOGGER.info('done reading sesame files\n')
    return samples
