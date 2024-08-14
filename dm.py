from annotations import Annotations
from utils import remove_probe_suffix
from sample_sheet import SampleSheet

import numpy as np
import logging
import pandas as pd
import pyranges as pr

from patsy import dmatrix
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)


def combine_pvalues(pvals):
    z_scores = norm.ppf(pvals)  # Convert p-values to z-scores
    combined_z = np.sum(z_scores) / np.sqrt(len(z_scores))  # Sum z-scores and normalize
    combined_pval = norm.cdf(combined_z)  # Convert back to p-value
    return combined_pval


def get_model_parameters(betas_values: np.array, design_matrix_p: pd.DataFrame, factor_names: list[str]) -> np.array:
    m0 = sm.OLS(betas_values, design_matrix_p).fit()
    results = [m0.f_pvalue]
    for factor in factor_names:
        results.extend([m0.tvalues[factor], m0.params[factor], m0.bse[factor]])
    return results


def get_dml(betas: pd.DataFrame, formula: str, sample_info: pd.DataFrame | SampleSheet) -> pd.DataFrame | None:
    """Get Differentially Methylated Locus
    `sample_info` must be either a pandas dataframe or a sample sheet. It must have the sample names in a column
        called `sample_names` and the column(s) used in the formula."""

    # check the input
    if isinstance(sample_info, SampleSheet):
        sample_info = sample_info.df
    elif not isinstance(sample_info, pd.DataFrame):
        LOGGER.error('get_dml() : parameter sample_info must be a dataframe or a sample sheet object')
        return None

    # data init.
    betas = betas.reset_index().set_index('probe_id')
    betas = betas.drop(columns=['type', 'channel', 'probe_type', 'index'], errors='ignore')

    # model
    sample_info = sample_info.set_index('sample_name')
    design_matrix = dmatrix(formula, sample_info, return_type='dataframe')
    factor_names = design_matrix.columns[1:]  # remove 'Intercept' from factors

    # output column names
    column_names = ['p_value']
    for factor in factor_names:
        column_names.extend([f't_value_{factor}', f'estimate_{factor}', f'std_err_{factor}'])

    # if it's a small dataset, don't parallelize
    if len(betas) <= 10000:
        result_array = [get_model_parameters(row[1:], design_matrix, factor_names) for row in betas.itertuples()]
    else:
        def wrapper_get_model_parameters(row):
            return get_model_parameters(row, design_matrix, factor_names)
        result_array = Parallel(n_jobs=-1)(delayed(wrapper_get_model_parameters)(row[1:]) for row in betas.itertuples())

    return pd.DataFrame(result_array, index=betas.index, columns=column_names)


def get_dmr(betas: pd.DataFrame, annotation: Annotations, dml: pd.DataFrame,
            dist_cutoff: float | None = None, seg_per_locus: float = 0.5) -> pd.DataFrame:
    """Get Differentially Methylated Regions (DMR) """

    # data init.
    betas = betas.reset_index().set_index('probe_id')
    betas = betas.drop(columns=['type', 'channel', 'probe_type', 'index'], errors='ignore')

    # get genomic range information (for chromosome id and probe position)
    probe_coords_df = annotation.get_genomic_ranges().drop(columns='Strand', errors='ignore')
    non_empty_coords_df = probe_coords_df[probe_coords_df.End > probe_coords_df.Start]  # remove 0-width ranges

    betas_no_na = betas.dropna()  # remove probes with missing values
    cpg_ids = non_empty_coords_df.join(betas_no_na, how='inner')

    # if there was no match, try again after trimming the suffix from the genomic ranges probe IDs
    if len(cpg_ids) == 0:
        non_empty_coords_df.index = non_empty_coords_df.index.map(remove_probe_suffix)
        cpg_ids = non_empty_coords_df.join(betas_no_na)

    if len(cpg_ids) == 0:
        LOGGER.error('No match found between genomic probe coordinates and beta probe IDs')
        return pd.DataFrame()

    # sort ranges and identify last probe of each chromosome
    # cpg_ranges = pr.PyRanges(cpg_ids).sort_ranges(natsorting=True)  # to have the same sorting as sesame
    cpg_ranges = pr.PyRanges(cpg_ids).sort_ranges()
    next_chromosome = cpg_ranges['Chromosome'].shift(-1)
    last_probe_in_chromosome = cpg_ranges['Chromosome'] != next_chromosome

    # compute Euclidian distance of beta values between two consecutive probes of each sample
    sample_names = betas.columns
    beta_euclidian_dist = (cpg_ranges[sample_names].diff(-1) ** 2).sum(axis=1)
    beta_euclidian_dist.iloc[-1] = None  # last probe shouldn't have a distance (default is 0 otherwise)

    # determine cut-off if not provided
    if dist_cutoff is None:
        # dist_cutoff = np.quantile(beta_euclidian_dist.dropna(), 1 - seg_per_locus)  # sesame (keep last probes)
        dist_cutoff = np.quantile(beta_euclidian_dist[~last_probe_in_chromosome], 1 - seg_per_locus)

    if dist_cutoff <= 0:
        LOGGER.warning(f'Euclidian distance cutoff for DMP should be > 0')
    LOGGER.info(f'Euclidian distance cutoff for DMP : {dist_cutoff}')
    LOGGER.info(f'Segments per locus : {seg_per_locus}')

    # find change points
    change_points = last_probe_in_chromosome | (beta_euclidian_dist > dist_cutoff)

    # give a unique ID to each segment
    segment_id = change_points.shift(fill_value=True).cumsum()
    segment_id.name = 'segment_id'

    # merging segments with all probes - including empty ones dropped at the beginning
    segments = probe_coords_df.loc[betas.index].join(segment_id).sort_values('segment_id')

    last_segment_id = segment_id.max()
    LOGGER.info(f'Number of segments : {last_segment_id}')

    # assign new segments IDs to NA segments
    na_segments_indexes = segments.segment_id.isna()
    nb_na_segments = na_segments_indexes.sum()
    if nb_na_segments > 0:
        LOGGER.info(f'Adding {nb_na_segments} NA segments')
        segments.loc[na_segments_indexes, 'segment_id'] = [n for n in range(nb_na_segments)] + last_segment_id + 1
        segments.segment_id = segments.segment_id.astype(int)

    # get each segment start and end
    segments_grouped = segments.groupby('segment_id')
    segments['segment_start'] = segments_grouped['Start'].transform('min')
    segments['segment_end'] = segments_grouped['End'].transform('max')

    # combine probes p values with segments
    combined = segments.join(dml)

    # compute values per segment
    grouped_seg = combined.groupby('segment_id')
    # todo fix factor en dur
    seg_est = pd.DataFrame({'segment_estimate': grouped_seg['estimate_sample_group[T.Sain]'].mean(),
                            'segment_p_value': grouped_seg['p_value'].apply(lambda p_values: combine_pvalues(p_values))})

    combined_values = combined.reset_index().set_index('segment_id').join(seg_est)
    combined_values['segment_p_value_adjusted'] = multipletests(combined_values['p_value'], method='fdr_bh')[1]
    combined_values = combined_values.reset_index().set_index('probe_id')

    return combined_values
