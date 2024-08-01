from detection import pOOBAH
from sample import Sample


def print_header(title: str) -> None:
    print('\n===================================================================')
    print(f'|  {title}')
    print('===================================================================\n')


def print_value(name, value) -> None:
    print(f'{name}\t\t:  {value}')


def print_pct(name, value) -> None:
    print(f'{name}\t\t:  {100*value} %')


def detection_stats(sample: Sample) -> None:
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