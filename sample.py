from pathlib import Path
import logging

import pandas as pd

from idat import IdatDataset

from annotations import Annotations, Channel
from sample_sheet import SampleSheet

LOGGER = logging.getLogger(__name__)


class Samples:

    def __init__(self, sheet: SampleSheet, annotation: Annotations):
        self.annotation = annotation
        self.sheet = sheet
        self.samples = {}

    def read_samples(self, datadir: str) -> None:

        for _, line in self.sheet.df.iterrows():
            sample = Sample()
            for channel in Channel:
                pattern = f'*{line.sentrix_id}*{line.sentrix_position}*{channel}*.idat'
                paths = [p.__str__() for p in Path(datadir).rglob(pattern)]
                if len(paths) == 0:
                    LOGGER.error(f'no paths found matching {pattern} in {datadir}')
                    continue
                if len(paths) > 1:
                    LOGGER.error(f'Too many files found matching {pattern} in {datadir} : {paths}')
                    continue
                LOGGER.info(f'reading file {paths[0]}')
                sample.set_idata(channel, IdatDataset(paths[0]))
            self.samples[line.sample_name] = sample

    def merge_manifest_info(self) -> None:
        for sample in self.samples.values():
            sample.merge_manifest_info(self.annotation)


class Sample:

    def __init__(self):
        self.idata = dict()
        self.df = None

    def set_idata(self, channel: Channel, dataset: IdatDataset) -> None:
        self.idata[channel] = dataset

    def merge_manifest_info(self, annotation: Annotations) -> None:
        channel_dfs = []
        for channel, idata in self.idata.items():
            manifest_df = annotation.manifest
            mask_df = annotation.mask
            idata_df = idata.probe_means

            sample_df = idata_df.join(manifest_df, how='inner')
            sample_df = sample_df.join(mask_df.drop(columns='probe_type'), on='probe_id', how='left')
            lost_probes = len(sample_df) - len(idata_df)
            if lost_probes > 0:
                print(f'Lost {lost_probes} while merging information with Manifest (out of {len(idata_df)})')

            # deduce methylation state (M = methylated, U = unmethylated) depending on infinium type
            sample_df['methylation_state'] = '?'
            sample_df.loc[sample_df.type == 'II', 'methylation_state'] = 'M' if channel.is_green else 'U'
            sample_df.loc[(sample_df.type == 'I') & (sample_df.index == sample_df.address_b), 'methylation_state'] = 'M'
            sample_df.loc[(sample_df.type == 'I') & (sample_df.index == sample_df.address_a), 'methylation_state'] = 'U'

            # don't know if this is useful
            sample_df['channel'] = channel.value
            channel_dfs.append(sample_df)
        self.df = pd.concat(channel_dfs)
