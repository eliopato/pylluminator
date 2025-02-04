"""Class that holds a collection of Sample objects and defines methods for batch processing."""
import os
import re
import gc
from importlib.resources.readers import MultiplexedPath
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF as ecdf

import pylluminator.sample_sheet as sample_sheet
from pylluminator.stats import norm_exp_convolution, quantile_normalization_using_target, background_correction_noob_fit
from pylluminator.stats import iqr
from pylluminator.utils import get_column_as_flat_array, set_channel_index_as, remove_probe_suffix
from pylluminator.utils import save_object, load_object, get_files_matching, get_logger, convert_to_path
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

    def __getitem__(self, item: int | str | list[str]) -> pd.DataFrame | None:
        if self._signal_df is not None:
            if isinstance(item, int) and item < len(self.sample_names):
                return self._signal_df[[self.sample_names[0]]].copy()
            elif isinstance(item, str) and item in self.sample_names:
                return self._signal_df[[item]].copy()
            elif isinstance(item, list):
                samples_names = []
                for sample in item:
                    if sample not in self.sample_names:
                        LOGGER.warning(f'Could not find sample "{sample}" in {self.sample_names}')
                    else:
                        samples_names.append(sample)
                if len(samples_names) > 0:
                    return self._signal_df[samples_names].copy()
            LOGGER.error(f'Could not find item {item} in {self.sample_names} of length {self.nb_samples}')
        else:
            LOGGER.error('No signal dataframe')
        return None

    ####################################################################################################################
    # Properties
    ####################################################################################################################

    @property
    def sample_names(self) -> list[str]:
        """Return the names of the samples contained in this object"""
        samples_in_sheet = set(self.sample_sheet.sample_name.values.tolist())
        samples_signal_df =  set(self._signal_df.columns.get_level_values(0))
        return list(samples_in_sheet & samples_signal_df)

    @property
    def nb_samples(self) -> int:
        """Count the number of samples contained in the object

        :return: number of samples
        :rtype: int"""
        return len(self.sample_names)

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

    def type1(self, mask: bool = True) -> pd.DataFrame:
        """Get the subset of Infinium type I probes, and apply the mask if `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        return self.get_signal_df(mask).xs('I', level='type', drop_level=False)

    def type2(self, mask: bool = True) -> pd.DataFrame:
        """Get the subset of Infinium type II probes, and apply the mask if `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        # idx = pd.IndexSlice
        type_ii_df = self.get_signal_df(mask).xs('II', level='type', drop_level=False)  # get only type II probes
        # type_ii_df = type_ii_df.loc[:, ((type_ii_df.columns.get_level_values('signal_channel') == 'R')
        #                                 & (type_ii_df.columns.get_level_values('methylation_state') == 'U'))
        #                                | ((type_ii_df.columns.get_level_values('signal_channel') == 'G')
        #                                   & (type_ii_df.columns.get_level_values('methylation_state') == 'M'))]
        type_ii_df = type_ii_df.loc[:, type_ii_df.columns.get_level_values('signal_channel').isin(['R', 'G'])]
        # return type_ii_df.loc[[idx[:, 'R', 'U'], idx[:, 'G', 'M']]]  # select non-NAN columns
        return type_ii_df.dropna(axis=1, how='all')
        # return type_ii_df

    def oob(self, mask: bool=True) -> pd.DataFrame | None:
        """Get the subset of out-of-band probes (for type I probes only), and apply the mask if `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame | None
        """
        return pd.concat([self.oob_green(mask), self.oob_red(mask)]).sort_index(axis=1)

    def oob_red(self, mask: bool = True) -> pd.DataFrame:
        """Get the subset of out-of-band red probes (for type I probes only), and apply the mask if `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        green_probes = self.get_signal_df(mask).xs('G', level='channel', drop_level=False)
        return green_probes.loc[:, (slice(None), 'R')]

    def oob_green(self, mask: bool = True) -> pd.DataFrame:
        """Get the subset of out-of-band green probes (for type I probes only), and apply the mask if `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        red_probes = self.get_signal_df(mask).xs('R', level='channel', drop_level=False)
        return red_probes.loc[:, (slice(None), 'G')]

    def ib_red(self, mask: bool = True) -> pd.DataFrame:
        """Get the subset of in-band red probes (for type I probes only), and apply the mask if `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        red_probes = self.get_signal_df(mask).xs('R', level='channel', drop_level=False)
        return red_probes.loc[:, (slice(None), 'R')]

    def ib_green(self, mask: bool = True) -> pd.DataFrame:
        """Get the subset of in-band green probes (for type I probes only), and apply the mask if `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        green_probes = self.get_signal_df(mask).xs('G', level='channel', drop_level=False)
        return green_probes.loc[:, (slice(None), 'G')]

    def ib(self, mask: bool = True) -> pd.DataFrame:
        """Get the subset of in-band probes (for type I probes only), and apply the mask if `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        return pd.concat([self.ib_red(mask), self.ib_green(mask)]).sort_index(axis=1)

    def type1_green(self, mask: bool = True) -> pd.DataFrame:
        """Get the subset of type I green probes, and apply the mask if `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        return self.get_signal_df(mask).xs( 'G', level='channel', drop_level=False)

    def type1_red(self, mask: bool = True) -> pd.DataFrame:
        """Get the subset of type I red probes, and apply the mask if `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        return self.get_signal_df(mask).xs('R', level='channel', drop_level=False)

    def meth(self, mask: bool = True) -> pd.DataFrame:
        """Get the subset of methylated probes, and apply the mask if `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        return self.get_signal_df(mask).xs('M', level='methylation_state', drop_level=False, axis=1)

    def unmeth(self, mask: bool = True) -> pd.DataFrame:
        """Get the subset of unmethylated probes, and apply the mask if `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        return self.get_signal_df(mask).xs('U', level='methylation_state', drop_level=False, axis=1)

    def cg_probes(self, mask: bool = True) -> pd.DataFrame:
        """Get CG (CpG) type probes, and apply the mask if `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        return self.get_probes_with_probe_type('cg', mask)

    def ch_probes(self, mask: bool = True) -> pd.DataFrame:
        """Get CH (CpH) type probes, and apply the mask if `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        return self.get_probes_with_probe_type('ch', mask)

    def snp_probes(self, mask: bool = True) -> pd.DataFrame:
        """Get SNP type probes ('rs' probes in manifest, but replaced by 'snp' when loaded), and apply the mask if
        `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        return self.get_probes_with_probe_type('snp', mask)

    def get_probes_with_probe_type(self, probe_type: str, mask: bool = True) -> pd.DataFrame:
        """Select probes by probe type, meaning e.g. CG, Control, SNP... (not infinium type I/II type), and apply the
        mask if `mask` is True

        :param probe_type: the type of probe to select (e.g. 'cg', 'snp'...)
        :type probe_type: str

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        if probe_type not in self.get_signal_df(mask).index.get_level_values('probe_type'):
            LOGGER.warning(f'no {probe_type} probes found')
            return pd.DataFrame()

        return self.get_signal_df(mask).xs(probe_type, level='probe_type', drop_level=False)#[['R', 'G']]

    def get_probes(self, probe_ids: list[str] | str, mask: bool = True) -> pd.DataFrame:
        """Returns the probes dataframe filtered on a list of probe IDs

        :param probe_ids: the IDs of the probes to select
        :type probe_ids: list[str]

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        if probe_ids is None or len(probe_ids) == 0:
            return pd.DataFrame()

        if isinstance(probe_ids, str):
            probe_ids = [probe_ids]

        probes_mask = self.get_signal_df(mask).index.get_level_values('probe_id').isin(probe_ids)
        return self.get_signal_df(mask)[probes_mask]


    ####################################################################################################################
    # Description, saving & loading
    ####################################################################################################################

    def __str__(self):
        return self.sample_names

    def __repr__(self):
        description = f'Samples object with {self.nb_samples} samples: {", ".join(self.sample_names)}\n'
        description += 'No annotation\n' if self.annotation is None else self.annotation.__repr__()
        description += f'{len(self._signal_df):,} probes'
        return description

    def copy(self):
        """Create a copy of the Samples object

        :return: a copy of the object
        :rtype: Samples"""
        new_samples = Samples(self.sample_sheet)
        new_samples.annotation = self.annotation
        new_samples.min_beads = self.min_beads
        new_samples.idata = None if self.idata is None else self.idata.copy()
        new_samples.masks =  None if self.masks is None else self.masks.copy()
        new_samples._signal_df = None if self._signal_df is None else self._signal_df.copy()
        return new_samples

    def save(self, filepath: str) -> None:
        """Save the current Samples object to `filepath`, as a pickle file

        :param filepath: path to the file to create
        :type filepath: str

        :return: None"""
        save_object(self, filepath)

    @staticmethod
    def load(filepath: str):
        """Load a pickled Samples object from `filepath`

        :param filepath: path to the file to read
        :type filepath: str

        :return: the loaded object"""
        return load_object(filepath, Samples)

    ####################################################################################################################
    # Data loading
    ####################################################################################################################

    def merge_annotation_info(self, annotation: Annotations, keep_idat=False, min_beads=1) -> None:
        """Merge manifest dataframe with probe signal values read from idat files to build the signal dataframe, adding
        channel information, methylation state and mask names for each probe.

        For manifest file, merging is done on `Illumina IDs`, contained in columns `address_a` and `address_b` of the
        manifest file.

        :param annotation: annotation data corresponding to the sample
        :type annotation: Annotations
        :param min_beads: filter probes with less than min_beads beads. Default: 1
        :type min_beads: int
        :param keep_idat: if set to True, keep idat data after merging the annotations. Default: False
        :type keep_idat: bool

        :return: None"""

        self.min_beads = min_beads

        # select probes signal values from idat dataframe, filtering by the minimum number of beads required
        probe_df_list = []
        for sample_name, channel_dict in self.idata.items():
            sample_dfs = []
            for channel, channel_df in channel_dict.items():
                df = channel_df.copy()
                df.loc[df.n_beads < min_beads, 'mean_value'] = pd.NA
                df['channel'] = str(channel)[0]
                df = df[['channel', 'mean_value']]
                df = df.reset_index().set_index(['illumina_id', 'channel'])
                df.columns = [sample_name]
                sample_dfs.append(df)
            probe_df_list.append(pd.concat(sample_dfs))

        probe_df = pd.concat(probe_df_list, axis=1)

        # auto detect annotation if not provided
        if annotation is None:
            probe_count = len(probe_df)  // 2 # nb of rows in the df = nb of rows in the idat file = nb of probes
            array_type = detect_array(probe_count)
            if array_type.is_human():
                annotation = Annotations(array_type, genome_version=GenomeVersion.HG38)
            else:
                annotation = Annotations(array_type, genome_version=GenomeVersion.MM39)

        self.annotation = annotation

        # prepare dataframes for merge
        indexes = ['type', 'channel', 'probe_type', 'probe_id', 'mask_info']
        probe_info = annotation.probe_infos[indexes + ['address_a', 'address_b']]
        probe_df = probe_df.reset_index().rename(columns={'channel': 'signal_channel'}).drop_duplicates()
        nb_probes_before_merge = len(probe_df)
        sample_df = pd.merge(probe_df, probe_info, how='inner', on='illumina_id')

        # check the number of probes lost in the merge
        lost_probes = nb_probes_before_merge - len(sample_df)
        pct_lost = 100 * lost_probes / nb_probes_before_merge
        LOGGER.info(f'Lost {lost_probes:,} illumina probes ({pct_lost:.2f}%) while merging information with Manifest')

        # deduce methylation state (M = methylated, U = unmethylated) depending on infinium type
        sample_df['methylation_state'] = '?'
        sample_df.loc[(sample_df.type == 'II') & (sample_df.signal_channel == 'G'), 'methylation_state'] = 'M'
        sample_df.loc[(sample_df.type == 'II') & (sample_df.signal_channel == 'R'), 'methylation_state'] = 'U'
        sample_df.loc[(sample_df.type == 'I') & (sample_df.illumina_id == sample_df.address_b), 'methylation_state'] = 'M'
        sample_df.loc[(sample_df.type == 'I') & (sample_df.illumina_id == sample_df.address_a), 'methylation_state'] = 'U'
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
        sample_df.columns = sample_df.columns.rename('sample_name', level=0)

        self._signal_df = sample_df

        for sample_name in self.sample_names:
            subset = sample_df[[(sample_name, 'G', 'M'), (sample_name, 'R', 'U')]]
            self.masks.add_mask(Mask(f'min_beads_{min_beads}', sample_name, subset.isna().any(axis=1)))

        # if we don't want to keep idata, make sure we free the memory
        if not keep_idat:
            self.idata = None
            gc.collect()

    ####################################################################################################################
    # Properties & getters
    ####################################################################################################################

    def get_signal_df(self, mask: bool = True) -> pd.DataFrame:
        """Get the methylation signal dataframe, and apply the mask if `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        sigdf = self._signal_df.copy()
        sample_names = self.sample_names

        if not isinstance(mask, bool):
            LOGGER.warning('Mask should be a boolean, setting it to True')
            mask = True

        if mask:
            # set probes to NA for all samples with the common mask
            common_mask = self.masks.get_mask()
            if common_mask is not None and len(common_mask) > 0:
                sigdf.loc[common_mask, sample_names] = None

            # then mask probes per sample if needed
            for sample_name in self.sample_names:
                sample_mask = self.masks.get_mask(sample_name=sample_name)
                if sample_mask is not None and len(sample_mask) > 0:
                    sigdf.loc[sample_mask, sample_name] = None

        return sigdf

    ####################################################################################################################
    # Mask functions
    ####################################################################################################################

    def apply_mask_by_names(self, names_to_mask: str, sample_name: str | None = None) -> None:
        """Match the names provided in `names_to_mask` with the probes mask info and mask these probes, adding them to
        the current mask if there is any.

        :param names_to_mask: can be a regex
        :type names_to_mask: str
        :param sample_name: The name of the sample to get masked indexes for. If None, returns masked indexes for all samples.
        :type sample_name: str | None

        :return: None"""

        if names_to_mask is None or len(names_to_mask) == 0:
            return

        new_mask = Mask(names_to_mask, sample_name, self._signal_df.mask_info.str.contains(names_to_mask))

        self.masks.add_mask(new_mask)

    def apply_quality_mask(self, sample_name: str | None = None) -> None:
        """Shortcut to apply quality mask on this sample

        :param sample_name: The name of the sample to mask. If None, mask indexes for all samples.
        :type sample_name: str | None
        :return: None"""
        self.apply_mask_by_names(self.annotation.quality_mask_names, sample_name)

    def apply_non_unique_mask(self, sample_name: str | None = None) -> None:
        """Shortcut to apply non-unique probes mask on this sample

        :param sample_name: The name of the sample to mask. If None, mask indexes for all samples.
        :type sample_name: str | None
        :return: None"""
        self.apply_mask_by_names(self.annotation.non_unique_mask_names, sample_name)

    def apply_xy_mask(self, sample_name: str | None = None) -> None:
        """Shortcut to mask probes from XY chromosome

        :param sample_name: The name of the sample to mask. If None, mask indexes for all samples.
        :type sample_name: str | None
        :return: None"""
        xy_probes_ids = self.annotation.probe_infos[self.annotation.probe_infos.chromosome.isin(['X', 'Y'])].probe_id
        xy_mask = pd.Series(self._signal_df.index.get_level_values('probe_id').isin(xy_probes_ids), self._signal_df.index)
        self.masks.add_mask(Mask('XY', sample_name, xy_mask))

    ####################################################################################################################
    # Control functions
    ####################################################################################################################

    def controls(self, mask: bool = True, pattern: str | None = None) -> pd.DataFrame | None:
        """Get the subset of control probes, matching the pattern with the probe_ids if a pattern is provided

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :param pattern: pattern to match against control probe IDs, case is ignored. Default: None
        :type pattern: str | None

        :return: methylation signal dataframe of the control probes, or None if None was found
        :rtype: pandas.DataFrame | None
        """
        control_df = self.get_probes_with_probe_type('ctl', mask)

        if control_df is None or len(control_df) == 0:
            LOGGER.info('No control probes found')
            return None

        if pattern is None:
            return control_df  # [['R', 'G']]

        probe_ids = control_df.index.get_level_values('probe_id')
        matched_ids = probe_ids.str.contains(pattern, flags=re.IGNORECASE)
        return control_df[matched_ids]  # [['R', 'G']]

    def get_normalization_controls(self, mask: bool = True, average=False) -> dict | pd.DataFrame | None:
        """Returns the control values to normalize green and red probes.

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :param average: if set to True, returns a dict with keys 'G' and 'R' containing the average of the control
            probes. Otherwise, returns a dataframe with selected probes. Default: False
        :type average: bool

        :return: the normalization controls as a dict or a dataframe, or None if None were found
        :rtype: dict | pandas.DataFrame | None
        """
        if self.controls(mask) is None:
            return None

        # patterns to find the probe IDs we need
        if self.annotation == ArrayType.HUMAN_27K:
            pattern_green = r'norm.green$'
            pattern_red = r'norm.red$'
        else:
            pattern_green = r'norm_c|norm_g$'
            pattern_red = r'norm_a|norm_t$'

        # find the red and green norm control probes according to their probe ID, and set the channel accordingly
        norm_green_df = self.controls(mask, pattern_green)
        norm_red_df = self.controls(mask, pattern_red)

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

    def get_negative_controls(self, mask: bool = True) -> pd.DataFrame | None:
        """Get negative control signal

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: the negative controls, or None if None were found
        :rtype: pandas.DataFrame | None
        """
        return self.controls(mask, 'negative')

    ####################################################################################################################
    # Channel functions
    ####################################################################################################################


    def infer_type1_channel(self, sample_name: str | None = None, switch_failed=False, mask_failed=False, summary_only=False) -> pd.DataFrame:
        """For Infinium type I probes, infer the channel from the signal values, setting it to the channel with the max
        signal. If max values are equals, the channel is set to R (as opposed to G in sesame).

        :param sample_name: the name of the sample to infer the channel for. If None, infer with all samples.
        :type sample_name: str | None
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

        # reset betas as we are modifying the signal dataframe
        self.reset_betas()
        sample_names = [sample_name] if isinstance(sample_name, str) else self.sample_names

        # subset to use
        type1_df = self._signal_df.loc['I', sample_names].droplevel('methylation_state', axis=1)

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
                probe_ids = self._signal_df.index.get_level_values('probe_id')
                failed_probe_ids = failed_idxs[failed_idxs].index.get_level_values('probe_id')
                mask_series = pd.Series(probe_ids.isin(failed_probe_ids), index=self._signal_df.index)
                self.masks.add_mask(Mask('failed_probes_inferTypeI', None, mask_series))

        # set the inferred channel as the new 'channel' index
        if not summary_only:
            # propagate the inferred channel to the signal dataframe
            self._signal_df.loc['I', 'inferred_channel'] = type1_df['inferred_channel'].values

            # propagate the inferred channel to the masks indexes
            for mask in self.masks.masks.values():
                mask_df = pd.concat([mask.series, self._signal_df['inferred_channel']], axis=1)
                mask_df = set_channel_index_as(mask_df, 'inferred_channel', drop=True)
                mask.series = mask_df.iloc[:, 0]

            self._signal_df = set_channel_index_as(self._signal_df, 'inferred_channel', drop=True)  # make the inferred channel the new channel index
            self._signal_df = self._signal_df.sort_index(axis=1)

        cols = ['channel', 'inferred_channel']
        return type1_df['inferred_channel'].reset_index().groupby(cols, observed=True).count()['probe_id']

    ####################################################################################################################
    # Preprocessing functions
    ####################################################################################################################

    def get_mean_ib_intensity(self, sample_name: str | None = None, mask=True) -> dict:
        """Computes the mean intensity of all the in-band measurements. This includes all Type-I in-band measurements
        and all Type-II probe measurements. Both methylated and unmethylated alleles are considered.

        :param sample_name: the name of the sample to get mean in-band intensity values for. If None, return mean
            in-band intensity values for every sample.
        :type sample_name: str | None
        :param mask: set to False if you don't want any mask to be applied. Default: True
        :type mask: bool

        :return: mean in-band intensity value
        :rtype: float """

        sample_names = [sample_name] if isinstance(sample_name, str) else self.sample_names

        sig_df = self.get_signal_df(mask)[sample_names]  # get signal dataframe for the sample(s)

        # set out-of-band signal to None
        sig_df.loc[(slice(None), 'G'), (slice(None), 'R')] = None
        sig_df.loc[(slice(None), 'R'), (slice(None), 'G')] = None

        mean_intensity = dict()
        for sample_name in sample_names:
            mean_intensity[sample_name] = sig_df[sample_name][['R', 'G']].mean(axis=None, skipna=True) #  masked rows stay NaN

        return mean_intensity

    def get_total_ib_intensity(self, sample_name: str | None = None, mask: bool = True) -> pd.DataFrame:
        """Computes the total intensity of all the in-band measurements. This includes all Type-I in-band measurements
        and all Type-II probe measurements. Both methylated and unmethylated alleles are considered.

        :param sample_name: the name of the sample to get total in-band intensity values for. If None, return total
            in-band intensity values for every sample.
        :type sample_name: str | None

        :param mask: set to False if you don't want any mask to be applied. Default: True
        :type mask: bool

        :return: the total in-band intensity values
        :rtype: pandas.DataFrame"""

        sample_names = [sample_name] if isinstance(sample_name, str) else self.sample_names

        series = []

        # sum the intensities for each probe (rows), but keep the columns separated by sample. Masked rows stay NaN
        for sample_name in sample_names:
            ib_red = self.ib_red(mask)[sample_name].sum(axis=1, min_count=1)
            ib_green = self.ib_green(mask)[sample_name].sum(axis=1, min_count=1)
            type2 = self.type2(mask)[sample_name].sum(axis=1, min_count=1)
            series.append(pd.concat([ib_red, ib_green, type2]))

        df = pd.concat(series, axis=1).sort_index()
        df.columns = sample_names
        return df

    def calculate_betas(self, include_out_of_band=False) -> None:
        """Calculate beta values for all probes. Values are stored in a dataframe and can be accessed via the betas()
        function

        :param include_out_of_band: is set to true, the Type 1 probes beta values will be calculated on
            in-band AND out-of-band signal values. If set to false, they will be calculated on in-band values only.
            equivalent to sumTypeI in sesame. Default: False
        :type include_out_of_band: bool

        :return: None"""

        df = self.get_signal_df(False).sort_index()
        idx = pd.IndexSlice
        # set NAs for Type II probes to 0, only where no methylation signal is expected
        df.loc['II', idx[:, 'R', 'M']] = 0
        df.loc['II', idx[:, 'G', 'U']] = 0
        # set out-of-band signal to 0 if the option include_out_of_band is not activated
        if not include_out_of_band:
            df.loc[idx['I', 'G'], idx[:, 'R']] = 0
            df.loc[idx['I', 'R'], idx[:, 'G']] = 0

        for sample_name in self.sample_names:
            sample_df = df[sample_name]
            # now we can calculate beta values
            methylated_signal = sample_df['R', 'M'] + sample_df['G', 'M']
            unmethylated_signal = sample_df['R', 'U'] + sample_df['G', 'U']

            # use clip function to set minimum values for each term as set in sesame
            beta_serie = methylated_signal.clip(lower=1) / (methylated_signal + unmethylated_signal).clip(lower=2)
            self._signal_df.loc[:, (sample_name, 'beta', '')] = beta_serie

        self._signal_df = self._signal_df.sort_index(axis=1)  # sort columns after adding beta values

    def reset_columns(self) -> None:
        """Remove all calculated columns (poobah p values, betas)

        :return None"""
        self._signal_df = self._signal_df.drop(['beta', 'p_value'], level='signal_channel', axis=1, errors='ignore')

    def reset_betas(self) -> None:
        """Remove beta columns from signal df

        :return: None"""
        self._signal_df = self._signal_df.drop('beta', level='signal_channel', axis=1, errors='ignore')

    def has_betas(self) -> bool:
        return 'beta' in self._signal_df.columns.get_level_values('signal_channel')

    def get_betas(self, sample_name: str | None = None, include_out_of_band = None, drop_na: bool = False,
                  custom_sheet: pd.DataFrame | None = None, mask: bool = False) -> pd.DataFrame | pd.Series | None:
        """Get the beta values for the sample. If no sample name is provided, return beta values for all samples.

        :param sample_name: the name of the sample to get beta values for. If None, return beta values for all samples.
        :type sample_name: str | None
        :param include_out_of_band: if set to True, calculate beta values on in-band AND out-of-band signal values. If
            set to False, calculate beta values on in-band values only. If not set, get the latest calculated betas if
            they already exist. Default: None
        :param drop_na: if set to True, drop rows with NA values. Default: False
        :type drop_na: bool
        :param custom_sheet: a custom sample sheet to filter samples. Ignored if sample_name is provided. Default: None
        :type custom_sheet: pandas.DataFrame | None
        :param mask: set to False if you don't want any mask to be applied. Default: False
        :type mask: bool

        :return: beta values as a DataFrame, or Series if sample_name is provided. If no beta values are found, return None
        :rtype: pandas.DataFrame | pandas.Series | None"""

        if include_out_of_band is not None:
            self.calculate_betas(include_out_of_band=include_out_of_band)
        elif not self.has_betas():
            self.calculate_betas()

        betas = self.get_signal_df(mask).xs('beta', level='signal_channel', axis=1).droplevel('methylation_state', axis=1)

        if custom_sheet is not None and sample_name is not None:
            LOGGER.warning('Both sample_name and custom_sheet are provided. Ignoring custom_sheet')
            custom_sheet = None

        if custom_sheet is not None:
            if len(custom_sheet) == 0:
                LOGGER.error('Empty custom_sheet')
                return None

            if 'sample_name' not in custom_sheet.columns:
                LOGGER.error(f'No sample_name column found in custom_sheet ({custom_sheet.columns})')
                return None

            # keep only samples that are both in sample sheet and beta columns
            filtered_samples = [col for col in custom_sheet.sample_name.values if col in betas.columns]
            if len(filtered_samples) == 0:
                LOGGER.error('No samples found')
                return None
            betas = betas[filtered_samples]

        if sample_name is not None:
            if sample_name not in betas.columns:
                LOGGER.error(f'Sample {sample_name} not found')
                return None
            betas = betas[sample_name]

        if drop_na:
            betas = betas.dropna()

        return betas

    def dye_bias_correction(self, sample_name: str | None = None, mask: bool = True, reference: dict | None = None) -> None:
        """Correct dye bias by linear scaling. Scale both the green and red signal to a reference level. If
        the reference level is not given, it is set to the mean intensity of all the in-band signals.

        :param sample_name: the name of the sample to correct dye bias for. If None, correct dye bias for all samples.
        :type sample_name: str | None

        :param mask: set to False if you don't want any mask to be applied. Default: True
        :type mask: bool

        :param reference: values to use as reference to scale red and green signal for each sample (=dict keys). Default: None
        :type: dict | None

        :return: None
        """

        self.reset_betas()  # reset betas as we are modifying the signal dataframe
        if sample_name is None:
            for sample_name in self.sample_names:
                self.dye_bias_correction(sample_name, mask, reference)
            return

        if not isinstance(sample_name, str):
            LOGGER.error('sample_name should be a string')
            return

        if sample_name not in self.sample_names:
            LOGGER.error(f'Sample {sample_name} not found')
            return

        if reference is None:
            reference = self.get_mean_ib_intensity(sample_name, mask)

        norm_values_dict = self.get_normalization_controls(mask, average=True)

        if norm_values_dict is None:
            return None

        for channel in ['R', 'G']:
            factor = reference[sample_name] / norm_values_dict[channel][sample_name]
            self._signal_df[(sample_name, channel)] *= factor

    def dye_bias_correction_l(self, sample_name: str | None = None, mask: bool = True, reference: dict | None = None) -> None:
        """Correct dye bias by linear scaling. Scale both the green and red signal to a reference level. If
        the reference level is not given, it is set to the mean intensity of all the in-band signals.

        :param sample_name: the name of the sample to correct dye bias for. If None, correct dye bias for all samples.
        :type sample_name: str | None

        :param mask: set to False if you don't want any mask to be applied. Default: True
        :type mask: bool

        :param reference: values to use as reference to scale red and green signal for each sample (=dict keys). Default: None
        :type: dict | None

        :return: None
        """

        self.reset_betas()  # reset betas as we are modifying the signal dataframe
        if sample_name is None:
            for sample_name in self.sample_names:
                self.dye_bias_correction_l(sample_name, mask, reference)
            return

        if not isinstance(sample_name, str):
            LOGGER.error('sample_name should be a string')
            return

        if sample_name not in self.sample_names:
            LOGGER.error(f'Sample {sample_name} not found')
            return

        if reference is None:
            reference = self.get_mean_ib_intensity(sample_name, mask)

        norm_values_dict = {'R': self.type1_red(mask)[(sample_name, 'R')].median(axis=None),
                            'G': self.type1_green(mask)[(sample_name, 'G')].median(axis=None)}

        for channel in ['R', 'G']:
            factor = reference[sample_name] / norm_values_dict[channel]
            self._signal_df[(sample_name, channel)] *= factor

    def dye_bias_correction_nl(self, sample_name: str | None = None, mask: bool = True) -> None:
        """Dye bias correction by matching green and red to mid-point.

        This function compares the Type-I Red probes and Type-I Grn probes and generates and mapping to correct signal
        of the two channels to the middle.

        :param sample_name: the name of the sample to correct dye bias for. If None, correct dye bias for all samples.
        :type sample_name: str | None

        :param mask: if True include masked probes in Infinium-I probes. No big difference is noted in practice. More
            probes are generally better. Default: True
        :type mask: bool

        :return: None
        """
        self.reset_betas()  # reset betas as we are modifying the signal dataframe
        sample_names = [sample_name] if isinstance(sample_name, str) else self.sample_names
        type_1_green = self.type1_green(mask)
        type_1_red = self.type1_red(mask)

        # check that there is not too much distortion between the two channels
        for sample_name in sample_names:
            total_intensity_type1 = self.get_total_ib_intensity(sample_name = sample_name, mask=False).loc['I', sample_name].sort_index()  # 0.5 sec

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
                type1_mask = pd.Series(self._signal_df.index.get_level_values('channel') == 'G', self._signal_df.index)
                self.masks.add_mask(Mask('dye bias nl', sample_name, type1_mask))
                return

            # all good, we can apply dye bias correction...
            sorted_intensities = {'G': np.sort(get_column_as_flat_array(type_1_green[sample_name], 'G', remove_na=True)),
                                  'R': np.sort(get_column_as_flat_array(type_1_red[sample_name], 'R', remove_na=True))}

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
                    LOGGER.warning(f'Max or min intensities are None. Aborting dye bias correction for sample {sample_name}.')
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

                self._signal_df.loc[:, [(sample_name, channel, 'M')]] = fit_function(self._signal_df[[(sample_name, channel, 'M')]].values)
                self._signal_df.loc[:, [(sample_name, channel, 'U')]] = fit_function(self._signal_df[[(sample_name, channel, 'U')]].values)

    def noob_background_correction(self, sample_name: str | None = None, mask: bool = True, use_negative_controls=True, offset=15) -> None:
        """Subtract the background for a sample.

        Background was modelled in a normal distribution and true signal in an exponential distribution. The Norm-Exp
        deconvolution is parameterized using Out-Of-Band (oob) probes. Multi-mapping probes are excluded.

        :param sample_name: the name of the sample to correct dye bias for. If None, correct dye bias for all samples.
        :type sample_name: str | None

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :param use_negative_controls: if True, the background will be calculated with both negative control and
            out-of-band probes. Default: True
        :type use_negative_controls: bool

        :param offset: A constant value to add to the corrected signal for padding. Default: 15
        :type offset: int | float

        :return: None
        """

        self.reset_betas()  # reset betas as we are modifying the signal dataframe
        sample_names = [sample_name] if isinstance(sample_name, str) else self.sample_names

        # mask non unique probes - saves previous mask to reset it afterwards
        initial_masks = self.masks.copy()
        if not mask:
            self.masks.reset_masks()
        self.apply_mask_by_names(self.annotation.non_unique_mask_names)

        # Background = out-of-band type 1 probes + (optionally) negative controls
        background_df = self.oob(True)
        if use_negative_controls:
            neg_controls = self.get_negative_controls(True)
            background_df = pd.concat([background_df, neg_controls])

        # Foreground = in-band type I probes + type 2 probes
        foreground_df = pd.concat([self.ib(True), self.type2(True)])

        # reset mask
        self.masks = initial_masks

        for sample_name in sample_names:

            bg = dict()
            fg = dict()

            for channel in ['R', 'G']:
                bg[channel] = get_column_as_flat_array(background_df[sample_name], channel, remove_na=True)
                fg[channel] = get_column_as_flat_array(foreground_df[sample_name], channel, remove_na=True)

                if len(bg[channel][bg[channel] > 0]) < 100:
                    LOGGER.warning('Not enough out of band signal to perform NOOB background subtraction')
                    return

                bg[channel][bg[channel] == 0] = 1
                fg[channel][fg[channel] == 0] = 1

                # cap at 10xIQR, this is to proof against multi-mapping probes
                bg[channel] = bg[channel][bg[channel] < (np.median(bg[channel]) + 10 * iqr(bg[channel]))]

                mu, sigma, alpha = background_correction_noob_fit(fg[channel], bg[channel])
                sample_df = self._signal_df[sample_name, channel]
                meth_corrected_signal = norm_exp_convolution(mu, sigma, alpha, sample_df['M'].values, offset)
                unmeth_corrected_signal = norm_exp_convolution(mu, sigma, alpha, sample_df['U'].values, offset)

                self._signal_df.loc[:, [[sample_name, channel, 'M']]] = meth_corrected_signal
                self._signal_df.loc[:, [[sample_name, channel, 'U']]] = unmeth_corrected_signal

    def scrub_background_correction(self, sample_name: str | None = None,  mask: bool = True) -> None:
        """Subtract residual background using background median.

        This function is meant to be used after noob.

        :param  sample_name: the name of the sample to scrub background for. If None, scrub background for all samples.
        :type sample_name: str | None
        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: None"""

        self.reset_betas()  # reset betas as we are modifying the signal dataframe
        sample_names = [sample_name] if isinstance(sample_name, str) else self.sample_names

        for sample_name in sample_names:

            median_bg = {'G': self.oob_green(mask)[sample_name].median(axis=None),
                         'R': self.oob_red(mask)[sample_name].median(axis=None)}

            for channel in ['G', 'R']:
                for methylation_state in ['U', 'M']:
                    idx = [(sample_name, channel, methylation_state)]
                    self._signal_df.loc[:, idx] = np.clip(self._signal_df[idx] - median_bg[channel], a_min=1, a_max=None)

    def poobah(self, sample_name: str | None = None, mask: bool = True, use_negative_controls=True, threshold=0.05) -> None:
        """Detection P-value based on empirical cumulative distribution function (ECDF) of out-of-band signal
        aka pOOBAH (p-vals by Out-Of-Band Array Hybridization).

        Adds two columns in the signal dataframe, 'p_value' and 'poobah_mask'. Add probes that are (strictly) above the
        defined threshold to the mask.

        :param sample_name: the name of the sample to use for the pOOBAH calculation. If None, use all samples.
        :type sample_name: str | None

        :param mask: True removes masked probes from background, False keeps them. Default: True
        :type mask: bool

        :param use_negative_controls: add negative controls as part of the background. Default True
        :type use_negative_controls: bool

        :param threshold: used to output a mask based on the p_values.
        :type threshold: float

        :return: None"""

        # mask non-unique probes - but first save previous mask to reset it afterward
        initial_masks = self.masks.copy()

        if not mask:
            self.masks.reset_masks()

        self.apply_mask_by_names(self.annotation.non_unique_mask_names)

        # Background = out-of-band type 1 probes + (optionally) negative controls
        background_df = self.oob(True)
        if use_negative_controls:
            neg_controls = self.get_negative_controls(True)
            background_df = pd.concat([background_df, neg_controls])

        # reset mask
        self.masks = initial_masks

        sample_names = [sample_name] if isinstance(sample_name, str) else self.sample_names

        for sample_name in sample_names:
            bg_green = get_column_as_flat_array(background_df[sample_name], 'G', remove_na=True)
            bg_red = get_column_as_flat_array(background_df[sample_name], 'R', remove_na=True)

            if np.sum(bg_red, where=~np.isnan(bg_red)) <= 100:
                LOGGER.debug('Not enough out of band signal, use empirical prior')
                bg_red = [n for n in range(1000)]

            if np.sum(bg_green, where=~np.isnan(bg_green)) <= 100:
                LOGGER.debug('Not enough out of band signal, use empirical prior')
                bg_green = [n for n in range(1000)]

            pval_green = 1 - ecdf(bg_green)(self._signal_df[(sample_name, 'G')].max(axis=1))
            pval_red = 1 - ecdf(bg_red)(self._signal_df[(sample_name, 'R')].max(axis=1))

            # set new columns with pOOBAH values
            p_value = np.min([pval_green, pval_red], axis=0)
            self._signal_df[(sample_name, 'p_value', '')] = p_value
            self._signal_df = self._signal_df.sort_index(axis=1)  # sort the columns after adding a new one

            # add a mask for the sample, depending on the threshold
            mask = self._signal_df[(sample_name, 'p_value', '')] >= threshold
            poobah_mask = Mask(f'poobah_{threshold}', sample_name, mask)
            self.masks.add_mask(poobah_mask)


def read_idata(sample_sheet_df: pd.DataFrame, datadir: str | Path) -> dict:
    """
    Reads IDAT files for each sample in the provided sample sheet, organizes the data by sample name and channel,
    and returns a dictionary with the IDAT data.

    :param sample_sheet_df: A DataFrame containing sample information, including columns for 'sample_name', 'sample_id',
        'sentrix_id', and 'sentrix_position'. Each row corresponds to a sample in the experiment.
    :type  sample_sheet_df: pandas.DataFrame

    :param datadir: The directory where the IDAT files are located.
    :type datadir: str

    :return: A dictionary where the keys are sample names (from the 'sample_name' column in `sample_sheet_df`), and the
        values are dictionaries mapping channel names (from `Channel`) to their respective IDAT data (as DataFrame
        objects, derived from the `IdatDataset` class).
    :rtype: dict

    Notes:
        - The function searches for IDAT files by sample ID and channel. If no files are found, it attempts to search
          using the Sentrix ID and position.
        - If multiple files match the search pattern, an error is logged.
        - If no matching files are found, an error is logged and the sample is skipped.

    Example:
        idata = read_idata(sample_sheet_df, '/path/to/data')
    """
    idata = {}

    for _, line in sample_sheet_df.iterrows():

        idata[line.sample_name] = dict()

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
            idata[line.sample_name][channel] = IdatDataset(idat_filepath).probes_df

    return idata


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

    # only load the N first samples
    if max_samples is not None:
        sample_sheet_df = sample_sheet_df.head(max_samples)

    samples = Samples(sample_sheet_df)
    samples.idata = read_idata(sample_sheet_df, datadir)
    samples._masked_indexes_per_sample = {sample_name: set() for sample_name in samples.idata.keys()}

    if samples.idata is None or len(samples.idata) == 0:
        LOGGER.error('no idat files found')
        return None

    samples.merge_annotation_info(annotation, keep_idat, min_beads)

    LOGGER.info('reading sample files done\n')
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
    sample_names = [f.stem.split('.csv')[0] for f in file_list]
    samples.sample_sheet = pd.DataFrame({'sample_id': sample_names, 'sample_name': sample_names})
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

        # merge manifest and mask
        sample_df = sig_df.join(manifest, how='inner')

        # move Green type II probes values to MG column
        sample_df.loc[sample_df.type == 'II', 'MG'] = sample_df.loc[sample_df.type == 'II', 'UG']
        sample_df.loc[sample_df.type == 'II', 'UG'] = np.nan

        # set signal channel for type II probes
        sample_df.loc[(sample_df.type == 'II') & (sample_df.MG.isna()), 'channel'] = 'R'
        sample_df.loc[(sample_df.type == 'II') & (sample_df.UR.isna()), 'channel'] = 'G'

        # make multi-index for rows and columns
        sample_df = sample_df.reset_index().set_index(['type', 'channel', 'probe_type', 'probe_id'])
        sample_df = sample_df.loc[:, ['UR', 'MR', 'MG', 'UG', 'mask', 'mask_info']]  # order columns
        sample_df.columns = pd.MultiIndex.from_tuples([('R', 'U'), ('R', 'M'), ('G', 'M'), ('G', 'U'), ('mask', ''), ('mask_info', '')])

        # set mask as specified in the input file, then drop the mask column
        samples.masks.add_mask(Mask('sesame', name, sample_df['mask']))
        sample_df = sample_df.sort_index(axis=1).drop(columns='mask')
        dfs.append(sample_df)

    if len(dfs) == 0:
        return None

    if len(dfs) != len(sample_names):
        LOGGER.warning(f'{len(dfs)} dfs != {len(sample_names)} samples names for {datadir}')

    samples._signal_df = pd.concat(dfs, axis=1, keys=sample_names)

    LOGGER.info('done reading sesame files\n')
    return samples
