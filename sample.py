from pathlib import Path
import logging
import pandas as pd
import numpy as np
import re

from idat import IdatDataset
from annotations import Annotations, Channel, ArrayType
from sample_sheet import SampleSheet
from stats import *
from utils import get_column_as_flat_array

LOGGER = logging.getLogger(__name__)


class Samples:

    def __init__(self, sheet: SampleSheet):
        self.annotation = None
        self.sheet = sheet
        self.samples = {}

    def read_samples(self, datadir: str, max_samples: int | None = None) -> None:
        """Search for idat files in the datadir through all sublevels. The idat files are supposed to match the
        information from the sample sheet and follow this naming convention:
        `[sentrix ID]*[sentrix position]*[channel].idat` where the `*` can be any characters. `channel` must be spelled
        `Red` or Grn`."""

        LOGGER.info(f'>> start reading sample files from {datadir}')

        for _, line in self.sheet.df.iterrows():
            sample = Sample(line.sample_name)
            for channel in Channel:
                pattern = f'*{line.sentrix_id}*{line.sentrix_position}*{channel}*.idat'
                paths = [p.__str__() for p in Path(datadir).rglob(pattern)]
                if len(paths) == 0:
                    LOGGER.error(f'no paths found matching {pattern}')
                    continue
                if len(paths) > 1:
                    LOGGER.error(f'Too many files found matching {pattern} : {paths}')
                    continue
                LOGGER.info(f'reading file {paths[0]}')
                sample.set_idata(channel, IdatDataset(paths[0]))
            self.samples[line.sample_name] = sample
            if max_samples is not None and len(self.samples) == max_samples:
                break

        LOGGER.info(f'reading sample files done\n')

    def merge_annotation_info(self, annotation: Annotations) -> None:
        """For all samples, call the function to merge manifest and mask information to the methylation signal data
        read from idat files."""

        self.annotation = annotation
        LOGGER.info(f'>> start merging manifest and sample data frames')
        for sample in self.samples.values():
            sample.merge_annotation_info(self.annotation)
        LOGGER.info(f'done merging manifest and sample data frames\n')

    def infer_type1_channel(self, switch_failed=False, mask_failed=False) -> None:
        LOGGER.info(f'>> start inferring probes')
        for sample in self.samples.values():
            sample.infer_type1_channel(switch_failed, mask_failed)
        LOGGER.info(f'done inferring probes\n')

    def get_betas(self) -> dict:
        betas_dict = {}
        LOGGER.info(f'>> start calculating betas')
        for name, sample in self.samples.items():
            betas_dict[name] = sample.get_betas()
        LOGGER.info(f'done calculating betas\n')
        return betas_dict


class Sample:

    ####################################################################################################################
    # Initialization
    ####################################################################################################################

    def __init__(self, name: str):
        self.idata = dict()
        self.df = None
        self.full_df = None
        self.name = name
        self.annotation = None
        self.which_index_channel = None
        self.indexes_not_masked = None

    def set_idata(self, channel: Channel, dataset: IdatDataset) -> None:
        self.idata[channel] = dataset

    def merge_annotation_info(self, annotation: Annotations) -> None:
        """ Merge manifest and mask dataframes to idat information to get the methylation signal dataframe, adding
        channel information, methylation state and mask names for each probe. For manifest file, merging is done on
        `Illumina ID`, contained in columns `address_a` and `address_b` of the manifest file. For the mask file, we use
        the `Probe_ID` to merge. We further use `Probe_ID` as an index throughout the code."""

        LOGGER.info(f'merging sample {self.name} with manifest {annotation}')
        self.annotation = annotation
        channel_dfs = []

        for channel, idata in self.idata.items():
            idata_df = idata.probe_means

            # merge manifest and mask
            sample_df = idata_df.join(annotation.manifest, how='inner')
            if annotation.mask is not None:
                sample_df = sample_df.join(annotation.mask.drop(columns='probe_type'), on='probe_id', how='left')

            # check for any lost probe when merge manifest (as it is an inner merge)
            lost_probes = len(sample_df) - len(idata_df)
            if lost_probes > 0:
                LOGGER.warning(f'Lost {lost_probes} while merging information with Manifest (out of {len(idata_df)})')

            # deduce methylation state (M = methylated, U = unmethylated) depending on infinium type
            sample_df['methylation_state'] = '?'
            sample_df.loc[sample_df.type == 'II', 'methylation_state'] = 'M' if channel.is_green else 'U'
            sample_df.loc[(sample_df.type == 'I') & (sample_df.index == sample_df.address_b), 'methylation_state'] = 'M'
            sample_df.loc[(sample_df.type == 'I') & (sample_df.index == sample_df.address_a), 'methylation_state'] = 'U'

            # set the signal channel to 'R' or 'G' depending on the channel defined in the idat filename
            sample_df['signal_channel'] = channel.value[0]

            # to improve readability
            sample_df.loc[sample_df.probe_type == 'rs', 'probe_type'] = 'snp'

            channel_dfs.append(sample_df)

        self.full_df = pd.concat(channel_dfs)
        # self.full_df['masked'] = False  # not sure if needed

        # reshape dataframe to have something resembling sesame data structure - one row per probe
        self.df = self.full_df.pivot(values='mean_value',
                                     columns=['signal_channel', 'methylation_state'],
                                     index=['type', 'channel', 'probe_type', 'probe_id', 'mask_info'])  # 'masked'

        # index column 'channel' corresponds by default to the manifest channel. But it could change by calling
        # 'infer_type_i_channel()' e.g., so we need to keep track of the manifest_channel in another column
        self.df['manifest_channel'] = self.df.index.get_level_values('channel').values

        # make mask_info a column, not an index - and set NaN values to empty string to allow string search on it
        self.df['mask_info'] = self.df.index.get_level_values('mask_info').fillna('').values
        self.df = self.df.reset_index(level='mask_info', drop=True)

    ####################################################################################################################
    # Properties & getters
    ####################################################################################################################

    @property
    def oob_red(self) -> pd.DataFrame:
        """Get the subset of out-of-band red probes (for type I probes only)"""
        return self.df_masked.xs('G', level='channel', drop_level=False)[['R']]

    @property
    def oob_green(self) -> pd.DataFrame:
        """Get the subset of out-of-band green probes (for type I probes only)"""
        return self.df_masked.xs('R', level='channel', drop_level=False)[['G']]

    @property
    def oob(self) -> pd.DataFrame:
        """Get the subset of out-of-band probes (for type I probes only)"""
        return pd.concat([self.oob_green, self.oob_red])

    @property
    def ib_red(self) -> pd.DataFrame:
        """Get the subset of in-band red probes (for type I probes only)"""
        return self.df_masked.xs('G', level='channel', drop_level=False)[['G']]

    @property
    def ib_green(self) -> pd.DataFrame:
        """Get the subset of in-band green probes (for type I probes only)"""
        return self.df_masked.xs('R', level='channel', drop_level=False)[['R']]

    @property
    def ib(self) -> pd.DataFrame:
        """Get the subset of in-band probes (for type I probes only)"""
        return pd.concat([self.ib_red, self.ib_green])

    @property
    def type1(self) -> pd.DataFrame:
        """Get the subset of Infinium type I probes"""
        return self.df_masked.xs('I', level='type', drop_level=False)

    @property
    def type2(self) -> pd.DataFrame:
        """Get the subset of Infinium type II probes"""
        return self.df_masked.xs('II', level='type', drop_level=False)[[['R', 'U'], ['G', 'M']]]

    # def get_probes(self, probe_ids: list[str]) -> pd.DataFrame | None:
    #     """Returns the probes dataframe filtered on a list of probe IDs"""
    #     if probe_ids is None or len(probe_ids) == 0:
    #         return None
    #
    #     idx = pd.IndexSlice
    #     return self.df.loc[idx[:, :, :, probe_ids], :]

    ####################################################################################################################
    # Mask functions
    ####################################################################################################################

    @property
    def df_masked(self) -> pd.DataFrame:
        """Apply current mask to the methylation signal dataframe"""
        if self.indexes_not_masked is None or self.nb_probes_masked == 0:
            return self.df
        else:
            LOGGER.debug(f'{self.nb_probes_masked} probes masked')
            return self.df.loc[self.indexes_not_masked]

    @property
    def nb_probes_masked(self) -> int:
        """Count the number of probes currently masked"""
        if self.indexes_not_masked is None:
            return 0
        return len(self.df) - len(self.indexes_not_masked)

    def reset_mask(self, names_to_mask: str | None = None):
        """Reset the mask to None (=no probe masked) and optionally set it to a new mask if `names_to_mask` is set"""
        self.indexes_not_masked = None
        if names_to_mask is not None:
            self.add_mask(names_to_mask)

    def add_mask(self, names_to_mask: str) -> None:
        """Match the names provided in `names_to_mask` with the probes mask info and mask these probes, adding them to
        the current mask if there is any.
        `names_to_mask` can be a regex"""

        if self.annotation.mask is None:
            LOGGER.warning('No mask is defined')
            return None

        nb_masked_before_add = self.nb_probes_masked
        to_mask = self.df_masked.mask_info.str.contains(names_to_mask)

        if len(to_mask) == 0:
            LOGGER.info(f'No new probes masked, {nb_masked_before_add} are already masked')
            return

        self.indexes_not_masked = self.df_masked[~to_mask].index

        LOGGER.info(f'Adding mask, now {self.nb_probes_masked} probes are masked ({nb_masked_before_add} previously)')

    ####################################################################################################################
    # Control functions
    ####################################################################################################################

    @property
    def controls(self) -> pd.DataFrame | None:
        """Get the subset of control probes"""
        if 'ctl' not in self.df.index.get_level_values('probe_type'):
            LOGGER.warning('no control probes found')
            return None

        return self.df_masked.xs('ctl', level='probe_type', drop_level=False)[['R', 'G']]

    def get_control_probes_indexes(self, pattern: str):
        probe_ids = self.controls.index.get_level_values('probe_id')
        return probe_ids.str.contains(pattern, flags=re.IGNORECASE)

    def get_normalization_controls(self, average=False) -> dict | pd.DataFrame | None:
        """Returns the control values to normalize green and red probes. If `average=True`, returns a dict with keys 'G'
        and 'R' containing the average of the control probes. Otherwise, returns a dataframe with selected probes."""
        if self.controls is None:
            return None

        # patterns to find the probe IDs we need
        if self.annotation == ArrayType.HUMAN_27K:
            pattern_green = r'norm.green$'
            pattern_red = r'norm.red$'
        else:
            pattern_green = r'norm_c|norm_g$'
            pattern_red = r'norm_a|norm_t$'

        # find the red and green norm control probes according to their probe ID, and set the channel accordingly
        idx_green = self.get_control_probes_indexes(pattern_green)
        idx_red = self.get_control_probes_indexes(pattern_red)

        if len(idx_green) == 0 and len(idx_red) == 0:
            LOGGER.warning('No normalization control probes found for both channels')
            return None

        if average:
            return {'G': np.nanmean(self.controls.loc[idx_green, [['G', 'M']]]),
                    'R': np.nanmean(self.controls.loc[idx_red, [['R', 'U']]])}

        # make 'channel' index a column to modify it
        norm_controls = self.controls
        norm_controls = norm_controls.reset_index('channel', drop=False)

        # update channel information
        norm_controls.loc[idx_green, 'channel'] = 'G'
        norm_controls.loc[idx_red, 'channel'] = 'R'

        # put channel column back as an index, keeping the same level order
        levels_order = self.controls.index.names
        norm_controls = norm_controls.set_index('channel', append=True).reorder_levels(levels_order)

        # select control probes
        return norm_controls.loc[idx_green | idx_red].copy()

    def get_negative_controls(self) -> pd.DataFrame | None:
        """Get negative control signal"""
        if self.controls is None:
            return None

        idx_neg_ctrl = self.get_control_probes_indexes('negative')

        if len(idx_neg_ctrl) == 0:
            LOGGER.warning('No negative control probes found')
            return None

        return self.controls[idx_neg_ctrl]

    ####################################################################################################################
    # Channel functions
    ####################################################################################################################

    def set_channel_index_as(self, column: str, drop=True) -> None:
        """Use an existing column specified by argument `column` as the new `channel` index. To keep the column, set
        `drop` to False"""

        if column not in self.df.columns:
            LOGGER.error(f'column {column} not found in df ({self.df.columns})')
            return

        # save index levels order to keep the same index structure
        levels_order = self.df.index.names

        if 'channel' in self.df.columns and column != 'channel':
            LOGGER.warning('dropping existing column `channel`')
            self.df.drop(column=['channel'], inplace=True)

        if drop:
            self.df.rename(columns={column: 'channel'}, inplace=True)
        else:
            self.df['channel'] = self.df[column]  # copy values in a new column

        self.df = self.df.droplevel('channel').set_index('channel', append=True).reorder_levels(levels_order)

    def reset_channel_index(self) -> None:
        """Set the channel index as the manifest channel"""
        self.set_channel_index_as('manifest_channel', False)

    def infer_type1_channel(self, switch_failed=False, mask_failed=False) -> None:
        """For Infinium type I probes, infer the channel from the signal values, setting it to the channel with the max
        signal. If max values are equals, the channel is set to R (as opposed to G in sesame).

        switch_failed: if set to True, probes with NA values or whose max values are under a threshold (the 95th
        percentile of the background signals) will be switched back to their original value

        mask_failed: mask failed probes (same probes as switch_failed)"""

        # subset to use
        type1_df = self.df.loc['I'].droplevel('methylation_state', axis=1).copy()

        # get the channel (provided by the index) where the signal is at its max for each probe
        type1_df['inferred_channel'] = type1_df.idxmax(axis=1, numeric_only=True).values

        # handle failed probes
        if not switch_failed or mask_failed:
            # calculate background for type I probes
            bg_signal_values = np.concatenate([type1_df.loc[type1_df.inferred_channel == 'R', 'G'],
                                               type1_df.loc[type1_df.inferred_channel == 'G', 'R']])
            bg_max = np.percentile(bg_signal_values, 95)
            failed_idxs = (type1_df.max(axis=1, numeric_only=True) < bg_max) | (type1_df.isna().any(axis=1))
            LOGGER.info(f'number of failed probes switched back: \n{failed_idxs.groupby("channel").sum()}')

            if not switch_failed:
                # reset color channel to the value of 'manifest_channel' for failed indexes of type I probes
                failed_idxs_manifest_values = type1_df.loc[failed_idxs, 'manifest_channel'].tolist()
                type1_df.loc[failed_idxs, 'inferred_channel'] = failed_idxs_manifest_values
            # todo
            # if mask_failed:
            #     sdf.loc[inf1_idx[idx], 'mask'] = True

        LOGGER.info(f"summary: \n{type1_df.groupby(['manifest_channel', 'inferred_channel']).count().max(axis=1)}")

        self.df.loc['I', 'inferred_channel'] = type1_df['inferred_channel'].values

        # make the inferred channel the new channel index
        self.set_channel_index_as('inferred_channel', drop=True)

    ####################################################################################################################
    # Preprocessing functions
    ####################################################################################################################

    def get_mean_intensity(self) -> float:
        """Computes the mean intensity of all the in-band measurements. This includes all Type-I in-band measurements
        and all Type-II probe measurements. Both methylated and unmethylated alleles are considered."""
        return np.nanmean(np.concatenate([self.ib_red, self.ib_green, self.type2]))

    def get_betas(self) -> pd.Series:
        # todo check why sesame gives the option to calculate separately R/G channels for Type I probes (sum.TypeI arg)
        # https://github.com/zwdzwd/sesame/blob/261e811c5adf3ec4ecc30cdf927b9dcbb2e920b6/R/sesame.R#L191
        df = self.df.fillna(0)
        methylated_signal = df['R', 'M'] + df['G', 'M']
        unmethylated_signal = df['R', 'U'] + df['G', 'U']
        # use clip function to set minimum values for each term as set in sesame
        return methylated_signal.clip(lower=1) / (methylated_signal + unmethylated_signal).clip(lower=2)

    def dye_bias_correction(self, reference: float | None = None):
        """ Correct dye bias in by linear scaling. Scale both the green and red signal to a reference (ref) level. If
        the reference level is not given, it is set to the mean intensity of all the in-band signals."""

        if reference is None:
            reference = self.get_mean_intensity()

        norm_values_dict = self.get_normalization_controls(average=True)

        for channel in ['R', 'G']:
            factor = reference / norm_values_dict[channel]
            self.df[channel] = self.df[channel] * factor

    def apply_noob_background_correction(self, use_negative_controls=True, offset=15) -> None:
        """
        Subtract the background. Background was modelled in a normal distribution and true signal in an exponential
        distribution. The Norm-Exp deconvolution is parameterized using Out-Of-Band (oob) probes. Multi-mapping probes
        are excluded.
        When `use_negative_controls = True`, background will be calculated ib both negative control and out-of-band probes
        """
        # mask non unique probes - saves previous mask to reset it afterwards
        previous_unmasked_indexes = self.indexes_not_masked
        self.add_mask(self.annotation.non_unique_mask_names)

        # Background = out-of-band type 1 probes + (optionally) negative controls
        background_df = self.oob
        if use_negative_controls:
            neg_controls = self.get_negative_controls()
            background_df = pd.concat([background_df, neg_controls])

        # Foreground = in-band type I probes + type 2 probes
        foreground_df = pd.concat([self.ib, self.type2])

        # reset mask
        self.indexes_not_masked = previous_unmasked_indexes

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
            meth_corrected_signal = norm_exp_convolution(mu, sigma, alpha, self.df[channel]['M'].values, offset)
            unmeth_corrected_signal = norm_exp_convolution(mu, sigma, alpha, self.df[channel]['U'].values, offset)

            self.df.loc[:, [[channel, 'M']]] = meth_corrected_signal
            self.df.loc[:, [[channel, 'U']]] = unmeth_corrected_signal
