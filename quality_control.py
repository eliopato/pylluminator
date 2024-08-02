import pandas as pd

from detection import pOOBAH
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

    pvals_df = pOOBAH(sample, threshold=0.05)

    sample_probe_ids = sample.df.index.get_level_values('probe_id')
    manifest_probe_ids = sample.annotation.manifest.probe_id
    missing_from_manifest = len([probe for probe in manifest_probe_ids if probe not in sample_probe_ids])

    missing_pvalues = pvals_df['p_value'].isna().sum()

    value_missing = missing_pvalues + missing_from_manifest
    print_value('N. Probes w/ Missing Raw Intensity', value_missing)
    print_pct('% Probes w/ Missing Raw Intensity', value_missing / (len(pvals_df) + missing_from_manifest))

    pvals_df = pvals_df.dropna()

    value_detection = sum(pvals_df['mask'])
    print_value('N. Probes w/ Detection Success', value_detection)
    print_pct('% Detection Success', value_detection / len(pvals_df))

    if sample.indexes_not_masked is None:
        value_masked = value_detection
        pct_masked = value_detection / len(pvals_df)
    else:
        value_masked = sum(pvals_df.loc[sample.indexes_not_masked, 'mask'])
        pct_masked = value_detection / len(sample.indexes_not_masked)
    print_value('N. Probes w/ Detection Success (after masking)', value_masked)
    print_pct('% Detection Success (after masking)', pct_masked)

    for probe_type in ['cg', 'ch', 'snp']:
        probes = pvals_df.xs(probe_type, level='probe_type')
        probes_value = sum(probes['mask'])
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
