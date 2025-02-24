"""
Functions to give an insight on a single Sample object probes values by calculating and printing some
reference statistics.
"""
import numpy as np
import pandas as pd
from pylluminator.samples import Samples as Samples


def _print_header(title: str, apply_mask=False) -> None:
    """Format and print a QC section header

    :param title: title of the section header
    :type title: str
    :param apply_mask: True removes masked probes, False keeps them. Default False
    :type apply_mask: bool

    :return: None"""
    if apply_mask:
        mask_str = 'mask applied'
    else:
        mask_str = 'mask not applied'

    print('\n===================================================================')
    print(f'|  {title} - {mask_str}')
    print('===================================================================\n')


def _print_value(name: str, value) -> None:
    """Format and print a QC value

    :param name: name (description) of the value to display
    :type name: str
    :param value: value to display. Can be anything printable.

    :return: None"""
    if isinstance(value, (float, np.float32, np.float64)):
        print(f'{name:<55} {value:.2f}')
    elif isinstance(value, (int, np.int32, np.int64)):
        print(f'{name:<55} {value:,}')
    else:
        print(f'{name:<55} {value}')

def _print_pct(name: str, value) -> None:
    """Format and print a QC percentage (x100 will be applied to the input value)

    :param name: name (description) of the value to display
    :type name: str
    :param value: value to display. Can be anything numeric.

    :return: None"""
    print(f'{name:<55} {100*value:.2f} %')


def detection_stats(samples: Samples, sample_label: str, apply_mask=False) -> None:
    """Print detection statistics of the given sample.

    :param samples: Samples object containing the sample to check
    :type samples: Samples
    :param sample_label: name of the sample to print the stats of
    :type sample_label: str
    :param apply_mask: True removes masked probes, False keeps them. Default False
    :type apply_mask: bool

    :return: None"""
    _print_header('Detection', apply_mask)
    poobah_threshold = 0.05
    samples = samples.copy()  # we don't want changes (mask) from poobah to change the samples object

    samples.poobah(sample_label, apply_mask, True, threshold=poobah_threshold)
    p_values_df = samples.get_signal_df(apply_mask)[(sample_label, 'p_value')]

    sample_probe_ids = p_values_df.index.get_level_values('probe_id')
    manifest_probe_ids = set(samples.annotation.probe_infos.probe_id)
    missing_from_manifest = len([probe for probe in manifest_probe_ids if probe not in sample_probe_ids])

    missing_p_values = p_values_df.isna().sum()

    value_missing = missing_p_values + missing_from_manifest
    _print_value('N. Probes w/ Missing Raw Intensity', value_missing)
    _print_pct('% Probes w/ Missing Raw Intensity', value_missing / (len(p_values_df) + missing_from_manifest))

    p_values_df = p_values_df.dropna()

    value_detection = sum(p_values_df < poobah_threshold)
    _print_value('N. Probes w/ Detection Success', value_detection)
    _print_pct('% Detection Success', value_detection / len(p_values_df))

    for probe_type in ['cg', 'ch', 'snp']:
        if probe_type not in p_values_df.index.get_level_values('probe_type'):
            _print_value(f'\nN. {probe_type} probes', 0)
            continue
        probes = p_values_df.xs(probe_type, level='probe_type')
        probes_value = sum(probes < poobah_threshold)
        print()
        _print_value(f'N. {probe_type} probes', len(probes))
        _print_value(f'N. Probes w/ Detection Success {probe_type}', probes_value)
        _print_pct(f'% Detection Success {probe_type}', probes_value / len(probes))


def intensity_stats(samples: Samples, sample_label: str, apply_mask=False) -> None:
    """Print intensity statistics of the given sample.

    :param samples: samples to print the stats of
    :type samples: Samples
    :param sample_label: name of the sample to print the stats of
    :type sample_label: str

    :param apply_mask: True removes masked probes, False keeps them. Default False
    :type apply_mask: bool

    :return: None"""
    _print_header('Signal intensity', apply_mask)

    _print_value('Mean in-band signal intensity', samples.get_mean_ib_intensity(sample_label, apply_mask)[sample_label])
    _print_value('Mean in-band signal intensity (M+U)', samples.get_total_ib_intensity(sample_label, apply_mask).mean()[sample_label])
    _print_value('Mean in-band type II signal intensity ', samples.type2(apply_mask)[sample_label].mean(axis=None, skipna=True))
    _print_value('Mean in-band type I Red signal intensity ', samples.ib_red(apply_mask)[sample_label].mean(axis=None, skipna=True))
    _print_value('Mean in-band type I Green signal intensity ', samples.ib_green(apply_mask)[sample_label].mean(axis=None, skipna=True))
    _print_value('Mean out-of-band type I Red signal intensity ', samples.oob_red(apply_mask)[sample_label].mean(axis=None, skipna=True))
    _print_value('Mean out-of-band type I Green signal intensity ', samples.oob_green(apply_mask)[sample_label].mean(axis=None, skipna=True))

    type_i_m_na = pd.isna(samples.meth(apply_mask).loc['I', sample_label]).values.sum()
    type_ii_m_na = pd.isna(samples.meth(apply_mask).loc['II', sample_label]['G']).values.sum()
    _print_value('Number of NAs in Methylated signal', type_i_m_na + type_ii_m_na)
    type_i_u_na = pd.isna(samples.unmeth(apply_mask).loc['I', sample_label]).values.sum()
    type_ii_u_na = pd.isna(samples.unmeth(apply_mask).loc['II', sample_label]['R']).values.sum()
    _print_value('Number of NAs in Unmethylated signal', type_ii_u_na + type_i_u_na)
    _print_value('Number of NAs in Type 1 Red signal', samples.type1_red(apply_mask)[sample_label].isna().values.sum())
    _print_value('Number of NAs in Type 1 Green signal', samples.type1_green(apply_mask)[sample_label].isna().values.sum())
    _print_value('Number of NAs in Type 2 signal', samples.type2(apply_mask)[sample_label].isna().values.sum())
    print('-- note : these NA values don\'t count probes that don\'t appear in .idat files; these are only counted in '
          'the `Detection - missing raw intensity` QC line')


def nb_probes_stats(samples: Samples, sample_label: str, apply_mask=False) -> None:
    """Print probe counts per Infinium type and Probe type

    :param samples: samples to print the stats of
    :type samples: Samples
    :param sample_label: name of the sample to print the stats of
    :type sample_label: str
    :param apply_mask: True removes masked probes, False keeps them. Default False
    :type apply_mask: bool

    :return: None"""

    _print_header('Number of probes', apply_mask)

    _print_value('Total : ', len(samples.get_signal_df(apply_mask)[sample_label]))
    _print_value('Type II : ', len(samples.type2(apply_mask)[sample_label]))
    _print_value('Type I Green : ', len(samples.type1_green(apply_mask)[sample_label]))
    _print_value('Type I Red : ', len(samples.type1_red(apply_mask)[sample_label]))
    _print_value('CG : ', len(samples.cg_probes(apply_mask)[sample_label]))
    _print_value('CH : ', len(samples.ch_probes(apply_mask)[sample_label]))
    _print_value('SNP : ', len(samples.snp_probes(apply_mask)[sample_label]))


def type1_color_channels_stats(samples: Samples, sample_label : str) -> None:
    """Print channel switch counts for Infinium type I probes

    :param samples: Samples object
    :type samples: Samples

    :param sample_label: name of the sample to print the stats of
    :type sample_label: str

    :return: None"""

    _print_header(f'{sample_label} Type I color channel', False)

    summary_inferred_channels = samples.infer_type1_channel(sample_labels=sample_label, summary_only=True)
    _print_value('Green to Green : ', summary_inferred_channels['G']['G'])
    _print_value('Green to Red : ', summary_inferred_channels['G']['R'])
    _print_value('Red to Red : ', summary_inferred_channels['R']['R'])
    _print_value('Red to Green : ', summary_inferred_channels['R']['G'])


def dye_bias_stats(samples: Samples, sample_label: str, apply_mask=False) -> None:
    """Print dye bias stats for Infinium type I probes

    :param samples: samples to print the stats of
    :type samples: Samples
    :param sample_label: name of the sample to print the stats of
    :type sample_label: str
    :param apply_mask: True removes masked probes, False keeps them. Default False
    :type apply_mask: bool

    :return: None"""

    _print_header('Dye bias', apply_mask)

    total_intensity_type1 = samples.get_total_ib_intensity(sample_label, apply_mask).loc['I'].sort_index()

    median_red = total_intensity_type1.loc['R'].median(skipna=True)[sample_label]
    median_green = total_intensity_type1.loc['G'].median(skipna=True)[sample_label]
    _print_value('Median Inf type I red channel intensity', median_red)
    _print_value('Median Inf type I green channel intensity', median_green)

    top_20_median_red = total_intensity_type1.loc['R', sample_label].nlargest(20).median(skipna=True)
    top_20_median_green = total_intensity_type1.loc['G', sample_label].nlargest(20).median(skipna=True)
    _print_value('Median of top 20 Inf type I red channel intensity', top_20_median_red)
    _print_value('Median of top 20 Inf type I green channel intensity', top_20_median_green)

    _print_value('Ratio of Red-to-green median intensities', median_red / median_green)
    red_green_distortion = (top_20_median_red/top_20_median_green) / (median_red / median_green)
    _print_value('Ratio of top vs global Red-to-green median intensities', red_green_distortion)


def betas_stats(samples: Samples, sample_label: str, apply_mask=False) -> None:
    """Print beta values stats

    :param samples: samples to print the stats of
    :type samples: Samples
    :param sample_label: name of the sample to print the stats of
    :type sample_label: str
    :param apply_mask: True removes masked probes, False keeps them.  Default False
    :type apply_mask: bool

    :return: None"""

    _print_header('Betas', apply_mask)
    samples = samples.copy()   # we don't want changes from processing to change the samples object

    samples.dye_bias_correction_nl(sample_label, apply_mask)
    samples.noob_background_correction(sample_label, apply_mask)
    samples.poobah(sample_label, apply_mask)
    samples.calculate_betas()
    betas = samples.get_betas(sample_label, apply_mask) # get betas as a pd.Series
    if betas is None or len(betas) == 0:
        print('No betas to analyze')
        return

    _print_value('Mean', betas.mean(skipna=True))
    _print_value('Median', betas.median(skipna=True))
    nb_non_na = len(betas.dropna())
    _print_pct('Unmethylated fraction (beta values < 0.3)', len(betas[betas < 0.3]) / nb_non_na)
    _print_pct('Methylated fraction (beta values > 0.7)', len(betas[betas > 0.7]) / nb_non_na)
    nb_na = len(betas) - nb_non_na
    _print_value('Number of NAs', nb_na)
    _print_pct('Fraction of NAs', nb_na / len(betas))

    for probe_type in 'cg', 'ch', 'snp':
        print(f'------ {probe_type} probes ------')
        subset_df = betas.xs(probe_type, level='probe_type')
        _print_value('Mean', subset_df.mean(skipna=True))
        _print_value('Median', subset_df.median(skipna=True))
        nb_non_na = len(subset_df.dropna())
        _print_pct('Unmethylated fraction (beta values < 0.3)', len(subset_df[subset_df < 0.3]) / nb_non_na)
        _print_pct('Methylated fraction (beta values > 0.7)', len(subset_df[subset_df > 0.7]) / nb_non_na)
        nb_na = len(subset_df) - nb_non_na
        _print_value('Number of NAs', nb_na)
        _print_pct('Fraction of NAs', nb_na / len(subset_df))

