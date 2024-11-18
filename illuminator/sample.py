"""
The Sample class is the
"""
import gc
import re
import os
from pathlib import Path

import pandas as pd
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF as ecdf

from illuminator.read_idat import IdatDataset
from illuminator.annotations import Annotations, Channel, ArrayType
from illuminator.stats import norm_exp_convolution, quantile_normalization_using_target, background_correction_noob_fit
from illuminator.stats import iqr
from illuminator.utils import get_column_as_flat_array, mask_dataframe, save_object, load_object, remove_probe_suffix

from illuminator.utils import get_logger

LOGGER = get_logger()


class Sample:
    """This class holds all methylation information of a sample in a `signal_df` data frame, as well as the current
    mask information in `masked_indexes`, a multi-index of all the currently masked probes.

    :ivar idata: raw data as read from .idat files. Dictionary with Channel as keys
    :vartype idata: dict

    :ivar name: sample name used as identifier.
    :vartype name: str

    :ivar annotation: probes metadata. Default: None
    :vartype annotation: Annotations | None

    :ivar masked_indexes: list of indexes masked for this sample
    :vartype masked_indexes: pandas.MultiIndex

    :ivar min_beads: required minimum number of beads per probe
    :vartype min_beads: int | None
    """

    ####################################################################################################################
    # Initialization
    ####################################################################################################################

    def __init__(self, name: str):
        """Initialize the object, with only its name - every other attributes are set to None

        :param name: sample name
        :type name: str"""
        self.idata = None
        self._signal_df = None
        self._betas_df = None
        self.name = name
        self.annotation = None
        self.masked_indexes = None
        self.min_beads = None

    def set_idata(self, channel: Channel, dataset: IdatDataset) -> None:
        """Add idata dataset to the sample idat dictionary, for the channel key passed in the argument

        :param channel: channel corresponding to the dataset
        :type channel: Channel
        :param dataset: dataset with .idat data
        :type dataset: IdatDataset

        :return: None
        """
        if self.idata is None:
            self.idata = {channel: dataset}
        else:
            self.idata[channel] = dataset

    def merge_annotation_info(self, annotation: Annotations, min_beads=1, keep_idat=False) -> None:
        """Merge manifest and mask dataframes to idat information to get the methylation signal dataframe, adding
        channel information, methylation state and mask names for each probe.

        For manifest file, merging is done on `Illumina IDs`, contained in columns `address_a` and `address_b` of the
        manifest file. For the mask file, we use the `Probe IDs` to merge. We further use `Probe IDs` as an index
        throughout the code.

        :param annotation: annotation data corresponding to the sample
        :type annotation: Annotations
        :param min_beads: filter probes with less than min_beads beads. Default: 1
        :type min_beads: int
        :param keep_idat: if set to True, keep idat data after merging the annotations. Default: False
        :type keep_idat: bool

        :return: None"""

        LOGGER.info(f'merging sample {self.name} with manifest {annotation}')
        self.annotation = annotation
        self.min_beads = min_beads

        # pivot table column names
        indexes = ['type', 'channel', 'probe_type', 'probe_id', 'mask_info']
        probe_info = annotation.probe_infos[indexes + ['address_a', 'address_b']]

        def process_idat(channel, idata):
            # merge manifest, mask and gene information
            sample_df = pd.merge(idata.probes_df, probe_info, how='inner', on='illumina_id')
            if channel == Channel.RED:
                # check for any lost probe when merging the manifest (as it is an inner merge) - only for one channel,
                # as both channels have the same probes
                nb_probes_before_merge = len(idata.probes_df)
                lost_probes = nb_probes_before_merge - len(sample_df)
                pct_lost = 100 * lost_probes / nb_probes_before_merge
                # print('sample df', len(sample_df))
                LOGGER.info(f'Lost {lost_probes:,} illumina probes ({pct_lost:.2f}%) while merging information with Manifest')
            # deduce methylation state (M = methylated, U = unmethylated) depending on infinium type
            sample_df['methylation_state'] = '?'
            sample_df.loc[sample_df.type == 'II', 'methylation_state'] = 'M' if channel.is_green else 'U'
            sample_df.loc[(sample_df.type == 'I') & (sample_df.index == sample_df.address_b), 'methylation_state'] = 'M'
            sample_df.loc[(sample_df.type == 'I') & (sample_df.index == sample_df.address_a), 'methylation_state'] = 'U'

            # set the signal channel to 'R' or 'G' depending on the channel defined in the idat filename
            sample_df['signal_channel'] = channel.value[0]  # first letter of the channel name

            return sample_df.drop(columns=['address_a', 'address_b'])

        channels_df = pd.concat([process_idat(channel, idata) for channel, idata in self.idata.items()], ignore_index=True)
        # remove probes with not enough beads
        low_nb_beads = channels_df.n_beads < min_beads
        nb_filtered_probes = sum(low_nb_beads)
        pct_filtered = 100 * nb_filtered_probes / len(channels_df)
        LOGGER.info(f'Removing {nb_filtered_probes:,} probes ({pct_filtered:.2f}%) that have less than {min_beads} beads')
        channels_df = channels_df[~low_nb_beads]

        # reshape dataframe to have something resembling sesame data structure - one row per probe
        self._signal_df = channels_df.pivot(index=indexes, values='mean_value',
                                           columns=['signal_channel', 'methylation_state'])
        # index column 'channel' corresponds by default to the manifest channel. But it could change by calling
        # 'infer_type_i_channel()' e.g., so we need to keep track of the manifest_channel in another column
        self._signal_df['manifest_channel'] = self._signal_df.index.get_level_values('channel').values

        # make mask_info a column, not an index - and set NaN values to empty string to allow string search on it
        self._signal_df['mask_info'] = self._signal_df.index.get_level_values('mask_info').fillna('').values
        self._signal_df = self._signal_df.reset_index(level='mask_info', drop=True)

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
        if mask:
            return mask_dataframe(self._signal_df, self.masked_indexes)
        else:
            return self._signal_df

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
        return self.get_signal_df(mask).xs('II', level='type', drop_level=False)[[['R', 'U'], ['G', 'M']]]

    def oob_red(self, mask: bool = True) -> pd.DataFrame:
        """Get the subset of out-of-band red probes (for type I probes only), and apply the mask if `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        return self.get_signal_df(mask).xs(('I', 'G'), level=['type', 'channel'], drop_level=False)[['R']]

    def oob_green(self, mask: bool = True) -> pd.DataFrame:
        """Get the subset of out-of-band green probes (for type I probes only), and apply the mask if `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        return self.get_signal_df(mask).xs(('I', 'R'), level=['type', 'channel'], drop_level=False)[['G']]

    def ib_red(self, mask: bool = True) -> pd.DataFrame:
        """Get the subset of in-band red probes (for type I probes only), and apply the mask if `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        return self.get_signal_df(mask).xs('G', level='channel', drop_level=False)[['G']]

    def ib_green(self, mask: bool = True) -> pd.DataFrame:
        """Get the subset of in-band green probes (for type I probes only), and apply the mask if `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        return self.get_signal_df(mask).xs('R', level='channel', drop_level=False)[['R']]

    def ib(self, mask: bool = True) -> pd.DataFrame:
        """Get the subset of in-band probes (for type I probes only), and apply the mask if `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        return pd.concat([self.ib_red(mask), self.ib_green(mask)])

    def type1_green(self, mask: bool = True) -> pd.DataFrame:
        """Get the subset of type I green probes, and apply the mask if `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        return self.get_signal_df(mask).xs(('I', 'G'), level=['type', 'channel'], drop_level=False)

    def type1_red(self, mask: bool = True) -> pd.DataFrame:
        """Get the subset of type I red probes, and apply the mask if `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        return self.get_signal_df(mask).xs(('I', 'R'), level=['type', 'channel'], drop_level=False)

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

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool
        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        if probe_type not in self.get_signal_df(mask).index.get_level_values('probe_type'):
            LOGGER.warning(f'no {probe_type} probes found')
            return pd.DataFrame()

        return self.get_signal_df(mask).xs(probe_type, level='probe_type', drop_level=False)[['R', 'G']]

    def get_probes_with_probe_ids(self, probe_ids: list[str], mask: bool = True) -> pd.DataFrame | None:
        """Returns the probes dataframe filtered on a list of probe IDs

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool
        :return: methylation signal dataframe
        :rtype: pandas.DataFrame
        """
        if probe_ids is None or len(probe_ids) == 0:
            return None

        probes_mask = self.get_signal_df(mask).index.get_level_values('probe_id').isin(probe_ids)
        return self.get_signal_df(mask)[probes_mask]

    def oob(self, mask: bool = True, channel=None) -> pd.DataFrame | None:
        """Get the subset of out-of-band probes (for type I probes only), and apply the mask if `mask` is True

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :param channel: specify a channel to return probes from this channel only. 'R' for red or 'G' for green. Default
            to None (return both channels)
        :type channel: str | None

        :return: methylation signal dataframe
        :rtype: pandas.DataFrame | None
        """
        if channel is None:
            return pd.concat([self.oob_green(mask), self.oob_red(mask)])
        elif channel == 'R':
            return self.oob_red(mask)
        elif channel == 'G':
            return self.oob_green(mask)
        else:
            LOGGER.error(f'Unknown channel {channel}. Must be any of : None, R or G.')
            return None

    def betas(self, mask: bool = True)-> pd.DataFrame | None:
        """Return the betas dataframe of the sample, and applies the current mask if the parameter mask is set to True
        (default). Return None if the betas have not been calculated yet.

       :param mask: (bool, optional, default True) True removes masked probes from betas, False keeps them. Default: False
       :type mask: bool

       :return: the beta values ad a pandas DataFrame, or None
       :rtype: pandas.DataFrame | None"""
        if mask:
            return mask_dataframe(self._betas_df, self.masked_indexes)
        else:
            return self._betas_df

    ####################################################################################################################
    # Mask functions
    ####################################################################################################################

    @property
    def nb_probes_masked(self) -> int:
        """Count the number of probes currently masked

        :return: number of masked probes
        :rtype: int"""
        if self.masked_indexes is None:
            return 0
        return len(self.masked_indexes)

    def reset_mask(self, names_to_mask: str | None = None, quiet=False):
        """Reset the mask to None (=no probe masked) and optionally set it to a new mask if `names_to_mask` is set.

        :param names_to_mask: None or string with a list of names of probes to mask, separated by a pipe. Default: None
        :type names_to_mask: str | None

        :param quiet: if set to True, don't print INFO logs for this function. Default: False
        :type quiet: bool

        :return: None"""
        if not quiet:
            LOGGER.debug('Resetting mask')
        self.masked_indexes = None
        if names_to_mask is not None:
            self.mask_names(names_to_mask)

    def mask_names(self, names_to_mask: str, quiet=False) -> None:
        """Match the names provided in `names_to_mask` with the probes mask info and mask these probes, adding them to
        the current mask if there is any.

        :param names_to_mask: can be a regex
        :type names_to_mask: str
        :param quiet: if set to True, don't print INFO logs for this function. Default: False
        :type quiet: bool

        :return: None"""

        if 'mask_info' not in self.annotation.probe_infos.columns:
            LOGGER.warning('No mask is defined')
            return None

        nb_masked_before_add = self.nb_probes_masked
        masked_signal_df = self.get_signal_df(True)
        to_mask = masked_signal_df.mask_info.str.contains(names_to_mask)

        if len(to_mask) == 0:
            if not quiet:
                LOGGER.info(f'No new probes masked, {nb_masked_before_add} are already masked')
            return

        self.mask_indexes(to_mask[to_mask].index, quiet)

    def mask_indexes(self, indexes_to_mask: pd.MultiIndex, quiet=False) -> None:
        """Add a list of indexes to the current mask

        :param indexes_to_mask: list of indexes to mask
        :type indexes_to_mask: pandas.MultiIndex
        :param quiet: if set to True, don't print INFO logs for this function. Default: False
        :type quiet: bool

        :return: None"""
        nb_masked_before_add = self.nb_probes_masked
        # no indexes to mask, nothing to do
        if indexes_to_mask is None or len(indexes_to_mask) == 0:
            return
        # no previously masked indexes, just set the property
        elif self.masked_indexes is None or nb_masked_before_add == 0:
            self.masked_indexes = indexes_to_mask
        # previously existing masked indexes, append the new ones
        else:
            self.masked_indexes = self.masked_indexes.append(indexes_to_mask).drop_duplicates()
        if not quiet:
            LOGGER.info(f'masking {self.nb_probes_masked - nb_masked_before_add:,} probes')

    def apply_quality_mask(self) -> None:
        """Shortcut to apply quality mask on this sample
        :return: None"""
        self.mask_names(self.annotation.quality_mask_names)

    def apply_non_unique_mask(self) -> None:
        """Shortcut to apply non unique probes mask on this sample
        :return: None"""
        self.mask_names(self.annotation.non_unique_mask_names)

    def apply_xy_mask(self) -> None:
        """Shortcut to mask probes from XY chromosome
        :return: None"""
        xy_probes_ids = self.annotation.probe_infos[self.annotation.probe_infos.chromosome.isin(['X', 'Y'])].probe_id
        xy_probes_indexes = self.get_probes_with_probe_ids(xy_probes_ids).index
        self.mask_indexes(xy_probes_indexes)

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
            return control_df[['R', 'G']]

        probe_ids = control_df.index.get_level_values('probe_id')
        matched_ids = probe_ids.str.contains(pattern, flags=re.IGNORECASE)
        return control_df[matched_ids][['R', 'G']]

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
            return {'G': np.nanmean(norm_green_df[[['G', 'M']]]), 'R': np.nanmean(norm_red_df[[['R', 'U']]])}

        # make 'channel' index a column to modify it
        norm_controls = pd.concat([norm_green_df, norm_red_df])
        norm_controls = norm_controls.reset_index('channel', drop=False)

        # update channel information
        norm_controls.loc[norm_green_df.index, 'channel'] = 'G'
        norm_controls.loc[norm_red_df.index, 'channel'] = 'R'

        # put channel column back as an index, keeping the same level order
        levels_order = norm_green_df.index.names
        norm_controls = norm_controls.set_index('channel', append=True).reorder_levels(levels_order)

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

    def set_channel_index_as(self, column: str, drop=True) -> None:
        """Use an existing column specified by argument `column` as the new channel index. To keep the column, set
        `drop` to False

        :param column: name of the column to use as the new channel index
        :type column: str
        :param drop: if set to False, keep the column used for the new index. Default: True

        :return: None"""

        if column not in self._signal_df.columns:
            LOGGER.error(f'column {column} not found in df ({self._signal_df.columns})')
            return

        # save index levels order to keep the same index structure
        lvl_order = self._signal_df.index.names

        if 'channel' in self._signal_df.columns and column != 'channel':
            LOGGER.warning('dropping existing column `channel`')
            self._signal_df.drop(column=['channel'], inplace=True)

        if drop:
            self._signal_df.rename(columns={column: 'channel'}, inplace=True)
        else:
            self._signal_df['channel'] = self._signal_df[column]  # copy values in a new column

        self._signal_df = self._signal_df.droplevel('channel').set_index('channel', append=True).reorder_levels(lvl_order)

    def reset_channel_index(self) -> None:
        """Set the channel index as the manifest channel.

        :return: None"""
        self.set_channel_index_as('manifest_channel', False)

    def infer_type1_channel(self, switch_failed=False, mask_failed=False, summary_only=False) -> None | pd.DataFrame:
        """For Infinium type I probes, infer the channel from the signal values, setting it to the channel with the max
        signal. If max values are equals, the channel is set to R (as opposed to G in sesame).

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
        self._betas_df = None

        # subset to use
        type1_df = self._signal_df.loc['I'].droplevel('methylation_state', axis=1).copy()

        # get the channel (provided by the index) where the signal is at its max for each probe
        type1_df['inferred_channel'] = type1_df.idxmax(axis=1, numeric_only=True).values

        # handle failed probes
        if not switch_failed or mask_failed:
            # calculate background for type I probes
            bg_signal_values = np.concatenate([type1_df.loc[type1_df.inferred_channel == 'R', 'G'],
                                               type1_df.loc[type1_df.inferred_channel == 'G', 'R']])
            bg_max = np.percentile(bg_signal_values, 95)
            failed_idxs = (type1_df.max(axis=1, numeric_only=True) < bg_max) | (type1_df.isna().any(axis=1))
            nb_failed_idxs = failed_idxs.groupby("channel", observed=True).sum()
            LOGGER.debug(
                f'number of failed probes switched back: Green {nb_failed_idxs["G"]} - Red {nb_failed_idxs["R"]}')

            if not switch_failed:
                # reset color channel to the value of 'manifest_channel' for failed indexes of type I probes
                failed_idxs_manifest_values = type1_df.loc[failed_idxs, 'manifest_channel'].tolist()
                type1_df.loc[failed_idxs, 'inferred_channel'] = failed_idxs_manifest_values

            if mask_failed:
                LOGGER.debug('Masking failed probes')
                self.mask_indexes(failed_idxs)

        summary = type1_df.groupby(['manifest_channel', 'inferred_channel'], observed=False).count().max(axis=1)

        # set the inferred channel as the new 'channel' index
        if not summary_only:
            self._signal_df.loc['I', 'inferred_channel'] = type1_df['inferred_channel'].values
            # make the inferred channel the new channel index
            self.set_channel_index_as('inferred_channel', drop=True)
            LOGGER.debug(f"type 1 channel inference summary: R -> R: {summary['R']['R']} - R -> G: {summary['R']['G']} "
                         f"- G -> G: {summary['G']['G']} - G -> R: {summary['G']['R']}")

        return summary

    ####################################################################################################################
    # Preprocessing functions
    ####################################################################################################################

    def get_mean_ib_intensity(self, mask=True) -> float:
        """Computes the mean intensity of all the in-band measurements. This includes all Type-I in-band measurements
        and all Type-II probe measurements. Both methylated and unmethylated alleles are considered.

        :param mask: set to False if you don't want any mask to be applied. Default: True
        :type mask: bool

        :return: mean in-band intensity value
        :rtype: float """

        return np.nanmean(np.concatenate([self.ib_red(mask), self.ib_green(mask), self.type2(mask)]))

    def get_total_ib_intensity(self, mask: bool = True) -> pd.DataFrame:
        """Computes the total intensity of all the in-band measurements. This includes all Type-I in-band measurements
        and all Type-II probe measurements. Both methylated and unmethylated alleles are considered.

        :param mask: set to False if you don't want any mask to be applied. Default: True
        :type mask: bool

        :return: the total in-band intensity values
        :rtype: pandas.DataFrame"""

        return pd.concat([self.ib_red(mask).sum(axis=1), self.ib_green(mask).sum(axis=1), self.type2(mask).sum(axis=1)])

    def calculate_betas(self, include_out_of_band=False) -> None:
        """Calculate beta values for all probes. Values are stored in a dataframe and can be accessed via the betas()
        function

        :param include_out_of_band: is set to true, the Type 1 probes Beta values will be calculated on
            in-band AND out-of-band signal values. If set to false, they will be calculated on in-band values only.
            equivalent to sumTypeI in sesame. Default: False
        :type include_out_of_band: bool

        :return: None"""

        df = self.get_signal_df(False).sort_index()
        # set NAs for Type II probes to 0, only where no methylation signal is expected
        df.loc['II', [['R', 'M']]] = 0
        df.loc['II', [['G', 'U']]] = 0
        # set out-of-band signal to 0 if the option include_out_of_band is not activated
        if not include_out_of_band:
            idx = pd.IndexSlice
            df.loc[idx['I', 'G'], 'R'] = 0
            df.loc[idx['I', 'R'], 'G'] = 0
        # now we can calculate beta values
        methylated_signal = df['R', 'M'] + df['G', 'M']
        unmethylated_signal = df['R', 'U'] + df['G', 'U']

        # use clip function to set minimum values for each term as set in sesame
        beta_values = methylated_signal.clip(lower=1) / (methylated_signal + unmethylated_signal).clip(lower=2)

        self._betas_df = pd.DataFrame(beta_values, columns=[self.name])

    def dye_bias_correction(self, mask: bool = True, reference: float | None = None) -> None:
        """Correct dye bias in by linear scaling. Scale both the green and red signal to a reference (ref) level. If
        the reference level is not given, it is set to the mean intensity of all the in-band signals.

        :param mask: set to False if you don't want any mask to be applied. Default: True
        :type mask: bool

        :param reference: value to use as reference to scale red and green signal. Default: None
        :type: float | None

        :return: None
        """

        # reset betas as we are modifying the signal dataframe
        self._betas_df = None

        if reference is None:
            reference = self.get_mean_ib_intensity(mask)

        norm_values_dict = self.get_normalization_controls(mask, average=True)

        if norm_values_dict is None:
            return None

        for channel in ['R', 'G']:
            factor = reference / norm_values_dict[channel]
            self._signal_df[channel] = self._signal_df[channel] * factor

    def dye_bias_correction_nl(self, mask: bool = True) -> None:
        """Dye bias correction by matching green and red to mid-point.

        This function compares the Type-I Red probes and Type-I Grn probes and generates and mapping to correct signal
        of the two channels to the middle.

        :param mask: if True include masked probes in Infinium-I probes. No big difference is noted in practice. More
            probes are generally better. Default: True
        :type mask: bool

        :return: None
        """

        # reset betas as we are modifying the signal dataframe
        self._betas_df = None

        total_intensity_type1 = self.get_total_ib_intensity(False).loc['I']

        # check that there is not too much distortion between the two channels

        median_red = np.median(total_intensity_type1.loc['R'])
        median_green = np.median(total_intensity_type1.loc['G'])

        top_20_median_red = np.median(total_intensity_type1.loc['R'].nlargest(20))
        top_20_median_green = np.median(total_intensity_type1.loc['G'].nlargest(20))

        red_green_distortion = (top_20_median_red / top_20_median_green) / (median_red / median_green)

        if red_green_distortion is None or red_green_distortion > 10:
            LOGGER.debug(f'Red-Green distortion is too high ({red_green_distortion}. Masking green probes')
            self.mask_indexes(self.type1_green(True).index)
            return

        # all good, we can apply dye bias correction...

        sorted_intensities = {'G': np.sort(get_column_as_flat_array(self.type1_green(mask), 'G')),
                              'R': np.sort(get_column_as_flat_array(self.type1_red(mask), 'R'))}

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

            self._signal_df.loc[:, [[channel, 'M']]] = fit_function(self._signal_df[[[channel, 'M']]].values)
            self._signal_df.loc[:, [[channel, 'U']]] = fit_function(self._signal_df[[[channel, 'U']]].values)

    def noob_background_correction(self, mask: bool = True, use_negative_controls=True, offset=15) -> None:
        """Subtract the background.

        Background was modelled in a normal distribution and true signal in an exponential distribution. The Norm-Exp
        deconvolution is parameterized using Out-Of-Band (oob) probes. Multi-mapping probes are excluded.

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :param use_negative_controls: if True, the background will be calculated with both negative control and
            out-of-band probes. Default: True
        :type use_negative_controls: bool

        :param offset: A constant value to add to the corrected signal for padding. Default: 15
        :type offset: int | float

        :return: None
        """

        # reset betas as we are modifying the signal dataframe
        self._betas_df = None

        # mask non unique probes - saves previous mask to reset it afterwards
        previous_masked_indexes = None if self.masked_indexes is None else self.masked_indexes.copy()
        if not mask:
            self.reset_mask(quiet=True)
        self.mask_names(self.annotation.non_unique_mask_names, quiet=True)

        # Background = out-of-band type 1 probes + (optionally) negative controls
        background_df = self.oob(True)
        if use_negative_controls:
            neg_controls = self.get_negative_controls(True)
            background_df = pd.concat([background_df, neg_controls])

        # Foreground = in-band type I probes + type 2 probes
        foreground_df = pd.concat([self.ib(True), self.type2(True)])

        # reset mask
        self.masked_indexes = previous_masked_indexes

        bg = dict()
        fg = dict()

        for channel in ['R', 'G']:
            bg[channel] = get_column_as_flat_array(background_df, channel, remove_na=True)
            fg[channel] = get_column_as_flat_array(foreground_df, channel, remove_na=True)

            if len(bg[channel][bg[channel] > 0]) < 100:
                LOGGER.warning('Not enough out of band signal to perform NOOB background subtraction')
                return

            bg[channel][bg[channel] == 0] = 1
            fg[channel][fg[channel] == 0] = 1

            # cap at 10xIQR, this is to proof against multi-mapping probes
            bg[channel] = bg[channel][bg[channel] < (np.median(bg[channel]) + 10 * iqr(bg[channel]))]

            mu, sigma, alpha = background_correction_noob_fit(fg[channel], bg[channel])
            meth_corrected_signal = norm_exp_convolution(mu, sigma, alpha, self._signal_df[channel]['M'].values, offset)
            unmeth_corrected_signal = norm_exp_convolution(mu, sigma, alpha, self._signal_df[channel]['U'].values,
                                                           offset)

            self._signal_df.loc[:, [[channel, 'M']]] = meth_corrected_signal
            self._signal_df.loc[:, [[channel, 'U']]] = unmeth_corrected_signal

    def scrub_background_correction(self, mask: bool = True) -> None:
        """Subtract residual background using background median.

        This function is meant to be used after noob.

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :return: None"""

        # reset betas as we are modifying the signal dataframe
        self._betas_df = None

        median_bg = {'G': np.median(self.oob_green(mask)),
                     'R': np.median(self.oob_red(mask))}

        for channel in ['G', 'R']:
            for methylation_state in ['U', 'M']:
                idx = [[channel, methylation_state]]
                self._signal_df.loc[:, idx] = np.clip(self._signal_df[idx] - median_bg[channel], a_min=1, a_max=None)

    def poobah(self, mask: bool = True, use_negative_controls=True, threshold=0.05) -> None:
        """Detection P-value based on empirical cumulative distribution function (ECDF) of out-of-band signal
        aka pOOBAH (p-vals by Out-Of-Band Array Hybridization).

        Adds two columns in the signal dataframe, 'p_value' and 'poobah_mask'. Add probes that are (strictly) above the
        defined threshold to the mask.

        :param mask: True removes masked probes, False keeps them. Default: True
        :type mask: bool

        :param use_negative_controls: add negative controls as part of the background. Default True
        :type use_negative_controls: bool

        :param threshold: used to output a mask based on the p_values.
        :type threshold: float

        :return: None"""

        # reset betas as we are modifying the signal dataframe
        self._betas_df = None

        # mask non-unique probes - but first save previous mask to reset it afterward
        previous_masked_indexes = None if self.masked_indexes is None else self.masked_indexes.copy()
        if not mask:
            self.reset_mask(quiet=True)

        # quiet = true because we don't want to log the mask change as it will be reset at the end
        self.mask_names(self.annotation.non_unique_mask_names, quiet=True)

        # Background = out-of-band type 1 probes + (optionally) negative controls
        background_df = self.oob(True)
        if use_negative_controls:
            neg_controls = self.get_negative_controls(True)
            background_df = pd.concat([background_df, neg_controls])

        bg_green = get_column_as_flat_array(background_df, 'G', remove_na=True)
        bg_red = get_column_as_flat_array(background_df, 'R', remove_na=True)

        if np.sum(bg_red, where=~np.isnan(bg_red)) <= 100:
            LOGGER.debug('Not enough out of band signal, use empirical prior')
            bg_red = [n for n in range(1000)]

        if np.sum(bg_green, where=~np.isnan(bg_green)) <= 100:
            LOGGER.debug('Not enough out of band signal, use empirical prior')
            bg_green = [n for n in range(1000)]

        # reset mask
        self.masked_indexes = previous_masked_indexes

        pval_green = 1 - ecdf(bg_green)(self._signal_df[['G']].max(axis=1))
        pval_red = 1 - ecdf(bg_red)(self._signal_df[['R']].max(axis=1))

        # set new columns with pOOBAH values
        self._signal_df['p_value'] = np.min([pval_green, pval_red], axis=0)
        self._signal_df['poobah_mask'] = self._signal_df['p_value'] > threshold

        # add pOOBAH mask to masked indexes
        self.mask_indexes(self._signal_df.loc[self._signal_df['poobah_mask']].index)

    ####################################################################################################################
    # Description, saving & loading
    ####################################################################################################################

    def save(self, filepath: str):
        """Save the current Sample object to `filepath`, as a pickle file

        :param filepath: path to the file to create
        :type filepath: str

        :return: None"""
        save_object(self, filepath)

    @staticmethod
    def load(filepath: str):
        """Load a pickled Sample object from `filepath`

        :param filepath: path to the file to read
        :type filepath: str

        :return: None"""
        return load_object(filepath, Sample)

    @staticmethod
    def from_sesame(filepath: str | os.PathLike, annotation: Annotations, name: str | None = None):
        """Read a SigDF object from SeSAMe, saved in a .csv file, and convert it into a Sample object.

         :param filepath: file containing the sigdf to read
         :type filepath: str | os.PathLike

        :param annotation: annotation data corresponding to the sample
        :type annotation: Annotations

        :param name: sample name
        :type name: str

        :return: None"""
        filepath = os.path.expanduser(filepath)

        LOGGER.debug(f'read {filepath}')

        if name is None:
            name = Path(filepath).stem

        sample = Sample(name)
        sample.annotation = annotation

        # read input file
        sig_df = pd.read_csv(filepath, low_memory=False)
        sig_df = sig_df.rename(columns={'col': 'channel', 'Probe_ID': 'probe_id'})

        # prepare manifest for merge
        manifest = annotation.probe_infos.loc[:, ['probe_id', 'type', 'probe_type', 'channel']]
        manifest = manifest.rename(columns={'channel': 'manifest_channel'})
        # remove probe suffix from manifest as they are not in SigDF files
        manifest['probe_id_to_join'] = manifest.probe_id.apply(remove_probe_suffix)
        manifest = manifest.set_index('probe_id_to_join')
        sig_df = sig_df.set_index('probe_id')

        # merge manifest and mask
        sample_df = sig_df.join(manifest, how='inner').set_index('probe_id')

        # move Green type II probes values to MG column
        sample_df.loc[sample_df.type == 'II', 'MG'] = sample_df.loc[sample_df.type == 'II', 'UG']
        sample_df.loc[sample_df.type == 'II', 'UG'] = np.nan

        # set signal channel for type II probes
        sample_df.loc[(sample_df.type == 'II') & (sample_df.MG.isna()), 'channel'] = 'R'
        sample_df.loc[(sample_df.type == 'II') & (sample_df.UR.isna()), 'channel'] = 'G'

        # make multi-index for rows and columns
        sample_df = sample_df.reset_index().set_index(['type', 'channel', 'probe_type', 'probe_id'])
        sample_df = sample_df.loc[:, ['UR', 'MR', 'MG', 'UG', 'mask', 'manifest_channel', 'mask_info']]  # order columns
        sample_df.columns = pd.MultiIndex.from_tuples([('R', 'U'), ('R', 'M'), ('G', 'M'), ('G', 'U'),
                                                       ('mask', ''), ('manifest_channel', ''), ('mask_info', '')])

        # set mask as specified in the input file, then drop the mask column
        sample.mask_indexes(sample_df[sample_df['mask']].index)
        sample.signal_df = sample_df.drop(columns=('mask', ''))

        return sample

    def __str__(self):
        description = f'Sample {self.name} :\n'
        description += 'No annotation\n' if self.annotation is None else self.annotation.__repr__()

        if self._signal_df is None:
            if self.idata is None:
                description += 'No data\n'
            else:
                description += 'Probes raw data:\n'
                for channel, dataset in self.idata.items():
                    description += f'\nChannel {channel}:\n {dataset}\n'
        else:
            description += 'Methylation dataframe: \n'
            description += f'{self._signal_df.head(3)}\n'
        return description

    def __repr__(self):
        return self.__str__()