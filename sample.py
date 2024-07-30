from pathlib import Path
import logging
import pandas as pd
import numpy as np
import re

from idat import IdatDataset
from annotations import Annotations, Channel, ArrayType
from sample_sheet import SampleSheet

LOGGER = logging.getLogger(__name__)


class Samples:

    def __init__(self, sheet: SampleSheet, annotation: Annotations):
        self.annotation = annotation
        self.sheet = sheet
        self.samples = {}

    def read_samples(self, datadir: str, max_samples: int | None = None) -> None:

        LOGGER.info(f'>> start reading sample files from {datadir}')

        for _, line in self.sheet.df.iterrows():
            sample = Sample(line.sample_name, self.annotation)
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

    def merge_manifest_info(self) -> None:
        LOGGER.info(f'>> start merging manifest and sample data frames')
        for sample in self.samples.values():
            sample.merge_manifest_info(self.annotation)
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

    def __init__(self, name: str, annotation: Annotations):
        self.idata = dict()
        self.df = None
        self.full_df = None
        self.name = name
        self.annotation = annotation

    def set_idata(self, channel: Channel, dataset: IdatDataset) -> None:
        self.idata[channel] = dataset

    def merge_manifest_info(self, annotation: Annotations) -> None:
        """ Merge manifest and mask dataframes to idat information."""

        LOGGER.info(f'merging sample {self.name}')
        channel_dfs = []

        for channel, idata in self.idata.items():
            idata_df = idata.probe_means

            sample_df = idata_df.join(annotation.manifest, how='inner')
            sample_df = sample_df.join(annotation.mask.drop(columns='probe_type'), on='probe_id', how='left')

            lost_probes = len(sample_df) - len(idata_df)
            if lost_probes > 0:
                LOGGER.warning(f'Lost {lost_probes} while merging information with Manifest (out of {len(idata_df)})')

            # deduce methylation state (M = methylated, U = unmethylated) depending on infinium type
            sample_df['methylation_state'] = '?'
            sample_df.loc[sample_df.type == 'II', 'methylation_state'] = 'M' if channel.is_green else 'U'
            sample_df.loc[(sample_df.type == 'I') & (sample_df.index == sample_df.address_b), 'methylation_state'] = 'M'
            sample_df.loc[(sample_df.type == 'I') & (sample_df.index == sample_df.address_a), 'methylation_state'] = 'U'

            sample_df['signal_channel'] = channel.value[0]

            # to improve readability
            sample_df.loc[sample_df.probe_type == 'rs', 'probe_type'] = 'snp'

            channel_dfs.append(sample_df)

        self.full_df = pd.concat(channel_dfs)
        # reshape dataframe to have something resembling sesame data structure - one row per probe
        self.df = self.full_df.pivot(values='mean_value',
                                     columns=['signal_channel', 'methylation_state'],
                                     index=['type', 'channel', 'probe_type', 'probe_id'])

        # index column 'channel' corresponds by default to the manifest channel. But it could change by calling
        # 'infer_type_i_channel()' e.g., so we need to keep track of the manifest_channel in another column
        self.df['manifest_channel'] = self.df.index.get_level_values('channel').values

    def get_betas(self) -> pd.DataFrame:
        # todo check why sesame gives the option to calculate separately R/G channels for Type I probes (sum.TypeI arg)
        # https://github.com/zwdzwd/sesame/blob/261e811c5adf3ec4ecc30cdf927b9dcbb2e920b6/R/sesame.R#L191
        df = self.df.fillna(0)
        methylated_signal = df['R', 'M'] + df['G', 'M']
        unmethylated_signal = df['R', 'U'] + df['G', 'U']
        # use clip function to set minimum values for each term as set in sesame
        return methylated_signal.clip(lower=1) / (methylated_signal + unmethylated_signal).clip(lower=2)

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
                type1_df.loc[failed_idxs, 'inferred_channel'] = type1_df.loc[failed_idxs, 'manifest_channel'].tolist()
            # todo
            # if mask_failed:
            #     sdf.loc[inf1_idx[idx], 'mask'] = True

        LOGGER.info(f"summary: \n{type1_df.groupby(['manifest_channel', 'inferred_channel']).count().max(axis=1)}")

        self.df.loc['I', 'inferred_channel'] = type1_df['inferred_channel'].values

        # make the inferred channel the new channel index
        self.set_channel_index_as('inferred_channel', drop=True)

    def set_channel_index_as(self, column: str, drop=True):
        """Use an existing column specified by argument `column` as the new `channel` index. To keep the column, set
        `drop` to False"""

        if column not in self.df.columns:
            LOGGER.error(f'column {column} not found in df ({self.df.columns})')
            return

        if 'channel' in self.df.columns:
            LOGGER.warning('dropping existing column `channel`')
            self.df.drop(column=['channel'], inplace=True)

        if drop:
            self.df.rename(columns={column: 'channel'}, inplace=True)
        else:
            self.df['channel'] = self.df[column]  # copy values in a new column

        self.df = (self.df.droplevel('channel')
                          .set_index('channel', append=True)
                          .reorder_levels(['type', 'channel', 'probe_type', 'probe_id']))

    def reset_channel_index(self):
        """Set the channel index as the manifest channel"""
        self.set_channel_index_as('manifest_channel', False)

    @property
    def oob_red(self) -> pd.DataFrame:
        """Get the subset of out-of-band red probes (for type I probes only)"""
        return self.df.xs('G', level='channel')['R']

    @property
    def oob_green(self) -> pd.DataFrame:
        """Get the subset of out-of-band green probes (for type I probes only)"""
        return self.df.xs('R', level='channel')['G']

    @property
    def ib_red(self) -> pd.DataFrame:
        """Get the subset of in-band red probes (for type I probes only)"""
        return self.df.xs('G', level='channel')['G']

    @property
    def ib_green(self) -> pd.DataFrame:
        """Get the subset of in-band green probes (for type I probes only)"""
        return self.df.xs('R', level='channel')['R']

    @property
    def type1(self) -> pd.DataFrame:
        """Get the subset of in-band green probes (for type I probes only)"""
        return self.df.loc['I']

    @property
    def type2(self) -> pd.DataFrame:
        """Get the subset of in-band green probes (for type I probes only)"""
        return self.df.loc['II', [['R', 'U'], ['G', 'M']]]

    @property
    def controls(self) -> pd.DataFrame | None:
        """Get the subset of control probes"""
        if 'ctl' not in self.df.index.get_level_values('probe_type'):
            LOGGER.warning('no control probes found')
            return None

        return self.df.xs('ctl', level='probe_type')[['R', 'G']]

    def mean_intensity(self) -> float:
        """Computes the mean intensity of all the in-band measurements. This includes all Type-I in-band measurements
        and all Type-II probe measurements. Both methylated and unmethylated alleles are considered."""
        return np.nanmean(np.concatenate([self.ib_red, self.ib_green, self.type2]))

    def norm_controls(self, average=False) -> dict | pd.DataFrame | None:
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

        controls = self.controls.reset_index()
        # find the red and green norm control probes according to their probe ID, and set the channel accordingly
        idx_green = controls.probe_id.str.contains(pattern_green, flags=re.IGNORECASE)
        idx_red = controls.probe_id.str.contains(pattern_red, flags=re.IGNORECASE)
        controls.loc[idx_green, 'channel'] = 'G'
        controls.loc[idx_red, 'channel'] = 'R'

        if average:
            return {'G': np.nanmean(controls.loc[idx_green, [['G', 'M']]]),
                    'R': np.nanmean(controls.loc[idx_red, [['R', 'U']]])}
        else:
            # return a dataframe with the norm probes, and the multi index format as usual
            return controls.loc[idx_green | idx_red].set_index(['type', 'channel', 'probe_id'])

    def dye_bias_correction(self, reference: float | None = None):
        """ Correct dye bias in by linear scaling. Scale both the green and red signal to a reference (ref) level. If
        the reference level is not given, it is set to the mean intensity of all the in-band signals."""

        if reference is None:
            reference = self.mean_intensity()

        norm_values_dict = self.norm_controls(average=True)

        for channel in ['R', 'G']:
            factor = reference / norm_values_dict[channel]
            self.df[channel] = self.df[channel] * factor
