"""
Functions to give an insight on a single Sample object probes values by calculating and printing some
reference statistics.
"""
import pandas as pd
import numpy as np

from pylluminator.samples import Samples as Samples


def print_header(title: str, mask=False) -> None:
    """Format and print a QC section header

    :param title: title of the section header
    :type title: str
    :param mask: True removes masked probes, False keeps them. Default False
    :type mask: bool

    :return: None"""
    if mask:
        mask_str = 'mask applied'
    else:
        mask_str = 'mask not applied'

    print('\n===================================================================')
    print(f'|  {title} - {mask_str}')
    print('===================================================================\n')


def print_value(name: str, value) -> None:
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

def print_pct(name: str, value) -> None:
    """Format and print a QC percentage (x100 will be applied to the input value)

    :param name: name (description) of the value to display
    :type name: str
    :param value: value to display. Can be anything numeric.

    :return: None"""
    print(f'{name:<55} {100*value:.2f} %')


def detection_stats(samples: Samples, sample_name: str, mask=False) -> None:
    """Print detection statistics of the given sample.

    :param samples: Samples object containing the sample to check
    :type samples: Samples
    :param mask: True removes masked probes, False keeps them. Default False
    :type mask: bool

    :return: None"""
    print_header('Detection', mask)

    samples.poobah(mask, True, threshold=0.05)
    p_values_df = sample.get_signal_df(mask)[['p_value', 'poobah_mask']]

    sample_probe_ids = sample.get_signal_df(mask).index.get_level_values('probe_id')
    manifest_probe_ids = set(sample.annotation.probe_infos.probe_id)
    missing_from_manifest = len([probe for probe in manifest_probe_ids if probe not in sample_probe_ids])

    missing_p_values = p_values_df['p_value'].isna().sum()

    value_missing = missing_p_values + missing_from_manifest
    print_value('N. Probes w/ Missing Raw Intensity', value_missing)
    print_pct('% Probes w/ Missing Raw Intensity', value_missing / (len(p_values_df) + missing_from_manifest))

    p_values_df = p_values_df.dropna()

    value_detection = len(p_values_df[~p_values_df.poobah_mask])
    print_value('N. Probes w/ Detection Success', value_detection)
    print_pct('% Detection Success', value_detection / len(p_values_df))

    for probe_type in ['cg', 'ch', 'snp']:
        if probe_type not in p_values_df.index.get_level_values('probe_type'):
            print_value(f'\nN. {probe_type} probes', 0)
            continue
        probes = p_values_df.xs(probe_type, level='probe_type')
        probes_value = len(probes[~probes.poobah_mask])
        print()
        print_value(f'N. {probe_type} probes', len(probes))
        print_value(f'N. Probes w/ Detection Success {probe_type}', probes_value)
        print_pct(f'% Detection Success {probe_type}', probes_value / len(probes))


def intensity_stats(sample: Sample, mask=False) -> None:
    """Print intensity statistics of the given sample.

    :param sample: sample to print the stats of
    :type sample: Sample
    :param mask: True removes masked probes, False keeps them. Default False
    :type mask: bool

    :return: None"""
    print_header('Signal intensity', mask)

    print_value('Mean in-band signal intensity', sample.get_mean_ib_intensity(mask))
    print_value('Mean in-band signal intensity (M+U)', sample.get_total_ib_intensity(mask).mean())
    print_value('Mean in-band type II signal intensity ', sample.type2(mask).mean(axis=None))
    print_value('Mean in-band type I Red signal intensity ', sample.ib_red(mask).mean(axis=None))
    print_value('Mean in-band type I Green signal intensity ', sample.ib_green(mask).mean(axis=None))
    print_value('Mean out-of-band type I Red signal intensity ', sample.oob_green(mask).mean(axis=None))
    print_value('Mean out-of-band type I Green signal intensity ', sample.oob_red(mask).mean(axis=None))

    type_i_m_na = pd.isna(sample.meth(mask).loc['I']).values.sum()
    type_ii_m_na = pd.isna(sample.meth(mask).loc['II']['G']).values.sum()
    print_value('Number of NAs in Methylated signal', type_i_m_na + type_ii_m_na)
    type_i_u_na = pd.isna(sample.unmeth(mask).loc['I']).values.sum()
    type_ii_u_na = pd.isna(sample.unmeth(mask).loc['II']['R']).values.sum()
    print_value('Number of NAs in Unmethylated signal', type_ii_u_na + type_i_u_na)
    print_value('Number of NAs in Type 1 Red signal', sample.type1_red(mask).isna().values.sum())
    print_value('Number of NAs in Type 1 Green signal', sample.type1_green(mask).isna().values.sum())
    print_value('Number of NAs in Type 2 signal', sample.type2(mask).isna().values.sum())
    print('-- note : these NA values don\'t count probes that don\'t appear in .idat files; these are only counted in '
          'the `Detection - missing raw intensity` QC line')


def nb_probes_stats(sample: Sample, mask=False) -> None:
    """Print probe counts per Infinium type and Probe type

    :param sample: sample to print the stats of
    :type sample: Sample
    :param mask: True removes masked probes, False keeps them. Default False
    :type mask: bool

    :return: None"""

    print_header('Number of probes', mask)

    print_value('Total : ', len(sample.get_signal_df(mask)))
    print_value('Type II : ', len(sample.type2(mask)))
    print_value('Type I Green : ', len(sample.type1_green(mask)))
    print_value('Type I Red : ', len(sample.type1_red(mask)))
    print_value('CG : ', len(sample.cg_probes(mask)))
    print_value('CH : ', len(sample.ch_probes(mask)))
    print_value('SNP : ', len(sample.snp_probes(mask)))


def type1_color_channels_stats(sample: Sample) -> None:
    """Print channel switch counts for Infinium type I probes

    :param sample: sample to print the stats of
    :type sample: Sample

    :return: None"""

    print_header('Type I color channel', False)

    summary_inferred_channels = sample.infer_type1_channel(summary_only=True)
    print_value('Green to Green : ', summary_inferred_channels['G']['G'])
    print_value('Green to Red : ', summary_inferred_channels['G']['R'])
    print_value('Red to Red : ', summary_inferred_channels['R']['R'])
    print_value('Red to Green : ', summary_inferred_channels['R']['G'])


def dye_bias_stats(sample: Sample, mask=False) -> None:
    """Print dye bias stats for Infinium type I probes

    :param sample: sample to print the stats of
    :type sample: Sample
    :param mask: True removes masked probes, False keeps them. Default False
    :type mask: bool

    :return: None"""

    print_header('Dye bias', mask)

    total_intensity_type1 = sample.get_total_ib_intensity(mask).loc['I']

    median_red = np.median(total_intensity_type1.loc['R'])
    median_green = np.median(total_intensity_type1.loc['G'])
    print_value('Median Inf type I red channel intensity', median_red)
    print_value('Median Inf type I green channel intensity', median_green)

    top_20_median_red = np.median(total_intensity_type1.loc['R'].nlargest(20))
    top_20_median_green = np.median(total_intensity_type1.loc['G'].nlargest(20))
    print_value('Median of top 20 Inf type I red channel intensity', top_20_median_red)
    print_value('Median of top 20 Inf type I green channel intensity', top_20_median_green)

    print_value('Ratio of Red-to-green median intensities', median_red / median_green)
    red_green_distortion = (top_20_median_red/top_20_median_green) / (median_red / median_green)
    print_value('Ratio of top vs global Red-to-green median intensities', red_green_distortion)


def betas_stats(sample: Sample, mask=False) -> None:
    """Print beta values stats

    :param sample: sample to print the stats of
    :type sample: Sample
    :param mask: True removes masked probes, False keeps them.  Default False
    :type mask: bool

    :return: None"""

    print_header('Betas', mask)

    sample.dye_bias_correction_nl(mask)
    sample.noob_background_correction(mask)
    sample.poobah(mask)
    sample.calculate_betas()
    betas = sample.betas(mask)[sample.name]  # get betas as a pd.Series

    print_value('Mean', betas.mean())
    print_value('Median', betas.median())
    nb_non_na = len(betas.dropna())
    print_pct('Unmethylated fraction (beta values < 0.3)', len(betas[betas < 0.3])/nb_non_na)
    print_pct('Methylated fraction (beta values > 0.7)', len(betas[betas > 0.7])/nb_non_na)
    nb_na = len(betas) - nb_non_na
    print_value('Number of NAs', nb_na)
    print_pct('Fraction of NAs', nb_na / len(betas))

    for probe_type in 'cg', 'ch', 'snp':
        print(f'------ {probe_type} probes ------')
        subset_df = betas.xs(probe_type, level='probe_type')
        print_value('Mean', subset_df.mean())
        print_value('Median', subset_df.median())
        nb_non_na = len(subset_df.dropna())
        print_pct('Unmethylated fraction (beta values < 0.3)', len(subset_df[subset_df < 0.3])/nb_non_na)
        print_pct('Methylated fraction (beta values > 0.7)', len(subset_df[subset_df > 0.7])/nb_non_na)
        nb_na = len(subset_df) - nb_non_na
        print_value('Number of NAs', nb_na)
        print_pct('Fraction of NAs', nb_na / len(subset_df))

