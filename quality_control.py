import pandas as pd
import numpy as np

from sample import Sample


def print_header(title: str) -> None:
    """Format and print a QC section header"""
    print('\n===================================================================')
    print(f'|  {title}')
    print('===================================================================\n')


def print_value(name: str, value) -> None:
    """Format and print a QC value"""
    print(f'{name}\t\t:  {value}')


def print_pct(name: str, value) -> None:
    """Format and print a QC percentage (x100 will be applied to the input value)"""
    print(f'{name}\t\t:  {100*value} %')


def detection_stats(sample: Sample) -> None:
    """Print detection statistics of the given sample."""
    print_header('Detection')

    sample.poobah(sample, threshold=0.05)
    p_values_df = sample.df[['p_value', 'poobah_mask']]

    sample_probe_ids = sample.df.index.get_level_values('probe_id')
    manifest_probe_ids = sample.annotation.manifest.probe_id
    missing_from_manifest = len([probe for probe in manifest_probe_ids if probe not in sample_probe_ids])

    missing_p_values = p_values_df['p_value'].isna().sum()

    value_missing = missing_p_values + missing_from_manifest
    print_value('N. Probes w/ Missing Raw Intensity', value_missing)
    print_pct('% Probes w/ Missing Raw Intensity', value_missing / (len(p_values_df) + missing_from_manifest))

    p_values_df = p_values_df.dropna()

    value_detection = sum(p_values_df['poobah_mask'])
    print_value('N. Probes w/ Detection Success', value_detection)
    print_pct('% Detection Success', value_detection / len(p_values_df))

    if sample.indexes_not_masked is None:
        value_masked = value_detection
        pct_masked = value_detection / len(p_values_df)
    else:
        value_masked = sum(p_values_df.loc[sample.indexes_not_masked, 'poobah_mask'])
        pct_masked = value_detection / len(sample.indexes_not_masked)
    print_value('N. Probes w/ Detection Success (after masking)', value_masked)
    print_pct('% Detection Success (after masking)', pct_masked)

    for probe_type in ['cg', 'ch', 'snp']:
        probes = p_values_df.xs(probe_type, level='probe_type')
        probes_value = sum(probes['poobah_mask'])
        print_value(f'N. Probes w/ Detection Success {probe_type}', probes_value)
        print_pct(f'% Detection Success {probe_type}', probes_value / len(probes))


def intensity_stats(sample: Sample) -> None:
    """Print intensity statistics of the given sample."""
    print_header('Signal intensity')
    print_value('Mean in-band signal intensity (masked)', sample.get_mean_ib_intensity())
    print_value('Mean in-band signal intensity (M+U, not masked)', sample.get_total_ib_intensity().mean())
    print_value('Mean in-band type II signal intensity ', sample.type2.mean(axis=None))
    print_value('Mean in-band type I Red signal intensity ', sample.ib_red.mean(axis=None))
    print_value('Mean in-band type I Green signal intensity ', sample.ib_green.mean(axis=None))
    print_value('Mean out-of-band type I Red signal intensity ', sample.oob_green.mean(axis=None))
    print_value('Mean out-of-band type I Green signal intensity ', sample.oob_red.mean(axis=None))

    type_i_m_na = pd.isna(sample.meth.loc['I']).values.sum()
    type_ii_m_na = pd.isna(sample.meth.loc['II', 'G']).values.sum()
    print_value('Number of NAs in Methylated signal', type_i_m_na + type_ii_m_na)
    type_i_u_na = pd.isna(sample.unmeth.loc['I']).values.sum()
    type_ii_u_na = pd.isna(sample.unmeth.loc['II', 'R']).values.sum()
    print_value('Number of NAs in Unmethylated signal', type_ii_u_na + type_i_u_na)
    print_value('Number of NAs in Type 1 Red signal', sample.type1_red.isna().values.sum())
    print_value('Number of NAs in Type 1 Green signal', sample.type1_green.isna().values.sum())
    print_value('Number of NAs in Type 2 signal', sample.type2.isna().values.sum())
    print('-- note : these NA values don\'t count probes that don\'t appear in .idat files; these are only counted in '
          'the `Detection - missing raw intensity` QC line')


def nb_probes_stats(sample: Sample, mask=False) -> None:
    """Print probe counts per Infinium type and Probe type"""

    previous_mask = sample.indexes_not_masked

    if mask:
        print_header('Number of probes (mask applied)')
    else:
        print_header('Number of probes')
        sample.reset_mask()

    print_value('Total : ', len(sample.df))
    print_value('Type II : ', len(sample.type2))
    print_value('Type I Green : ', len(sample.type1_green))
    print_value('Type I Red : ', len(sample.type1_red))
    print_value('CG : ', len(sample.cg_probes))
    print_value('CH : ', len(sample.ch_probes))
    print_value('SNP : ', len(sample.snp_probes))

    sample.indexes_not_masked = previous_mask


def type1_color_channels_stats(sample: Sample, mask=False) -> None:
    """Print channel switch counts for Infinium type I probes"""

    previous_mask = sample.indexes_not_masked

    if mask:
        print_header('Type I color channel (mask applied)')
    else:
        print_header('Type I color channel (no mask applied)')
        sample.reset_mask()

    summary_inferred_channels = sample.infer_type1_channel(summary_only=True)
    print_value('Green to Green : ', summary_inferred_channels['G']['G'])
    print_value('Green to Red : ', summary_inferred_channels['G']['R'])
    print_value('Red to Red : ', summary_inferred_channels['R']['R'])
    print_value('Red to Green : ', summary_inferred_channels['R']['G'])

    sample.indexes_not_masked = previous_mask


def dye_bias_stats(sample: Sample, mask=False) -> None:
    """Print dye bias stats for Infinium type I probes"""
    previous_mask = sample.indexes_not_masked

    if mask:
        print_header('Dye bias (mask applied)')
    else:
        print_header('Dye bias (no mask applied)')
        sample.reset_mask()

    total_intensity_type1 = sample.get_total_ib_intensity().loc['I']

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
    sample.indexes_not_masked = previous_mask


def betas_stats(sample: Sample):
    """Print Betas stats"""

    print_header('Betas')
    sample.dye_bias_correction()  # todo apply NL dye bias correction instead of linear
    sample.apply_noob_background_correction()
    sample.poobah()
    betas_df = sample.get_betas()

    print_header('Beta')
    print_value('Mean', betas_df.mean())
    print_value('Median', betas_df.median())
    nb_non_na = len(betas_df.dropna())
    print_pct('Unmethylated fraction (beta < 0.3)', len(betas_df[betas_df < 0.3])/nb_non_na)
    print_pct('Methylated fraction (beta > 0.7)', len(betas_df[betas_df > 0.7])/nb_non_na)
    nb_na = len(betas_df) - nb_non_na
    print_value('Number of NAs', nb_na)
    print_pct('Fraction of NAs', nb_na / len(betas_df))

    for probe_type in 'cg', 'ch', 'snp':
        print(f'------ {probe_type} probes ------')  # new line
        subset_df = betas_df.xs(probe_type, level='probe_type')
        print_value('Mean', subset_df.mean())
        print_value('Median', subset_df.median())
        nb_non_na = len(subset_df.dropna())
        print_pct('Unmethylated fraction (beta < 0.3)', len(subset_df[subset_df < 0.3])/nb_non_na)
        print_pct('Methylated fraction (beta > 0.7)', len(subset_df[subset_df > 0.7])/nb_non_na)
        nb_na = len(subset_df) - nb_non_na
        print_value('Number of NAs', nb_na)
        print_pct('Fraction of NAs', nb_na / len(subset_df))

