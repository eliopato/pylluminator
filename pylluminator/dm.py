"""
Functions used to compute DMR (Differentially Methylated Regions) and DMP (Differentially Methylated Probes).
"""

import warnings

import numpy as np
import pandas as pd
import pyranges as pr

from patsy import dmatrix
from scipy.stats import combine_pvalues
from statsmodels.api import OLS
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

from enum import Enum, unique
from pylluminator.utils import remove_probe_suffix, set_level_as_index, get_logger, merge_alt_chromosomes
from pylluminator.samples import Samples
from pylluminator.stats import get_factors_from_formula
from pylluminator.utils import merge_series_values

LOGGER = get_logger()


def combine_p_values_stouffer(p_values: pd.Series) -> np.ndarray:
    """shortcut to scipy's function, using Stouffer method to combine p-values. Only return the combined p-value

    :param p_values: p-values to combine
    :type p_values: pandas.Series

    :return: numpy array of combined p-values
    :rtype: numpy.ndarray"""
    if len(p_values) == 1:
        return p_values.iloc[0]
    return combine_pvalues(p_values, method='stouffer')[1]


def _get_model_parameters(betas_values, design_matrix: pd.DataFrame, factor_names: list[str], groups: pd.Series | None = None) -> list[float]:
    """Create an Ordinary Least Square model for the beta values, using the design matrix provided, fit it and
    extract the required results for DMP detection (p-value, t-value, estimate, standard error).

    :param betas_values: beta values to fit
    :type betas_values: 1D numpy.array_like
    :param design_matrix: design matrix for the model
    :type design_matrix: pandas.DataFrame
    :param factor_names: factors used in the model
    :type factor_names: list[str]
    :param groups: series holding the replicates information. Default: None
    :type groups: pandas.Series | None

    :return: f statistics p-value, effect size, and for each factor: p-value, t-value, estimate, standard error
    :rtype: list[float]"""
    if np.isnan(betas_values).all():
        return [np.nan] * (2 + 4 * len(factor_names))

    fitted_model = None
    if groups is None:
        fitted_model = OLS(betas_values, design_matrix, missing='drop').fit()  # drop NA values
    else:
        with warnings.catch_warnings(action='ignore'):
            try:
                # manually drop NA values as the model doesn't seem to handle them properly
                betas_values = np.array(betas_values)
                missing_betas = pd.isna(betas_values)
                fitted_model = MixedLM(betas_values[~missing_betas], design_matrix[~missing_betas],  groups[~missing_betas]).fit()
            except np.linalg.LinAlgError:
                return [np.nan] * (2 + 4 * len(factor_names))

    if fitted_model is None:
        return [np.nan] * (2 + 4 * len(factor_names))

    # fitted ols is a statsmodels.regression.linear_model.RegressionResultsWrapper (if OLS)
    # or statsmodels.regression.mixed_linear_model.MixedLMResults object
    estimates = fitted_model.params.iloc[1:].tolist()  + [0]  # remove the intercept
    effect_size = max(estimates) - min(estimates)
    if groups is None:
        results = [fitted_model.f_pvalue , effect_size]  # p-value of the F-statistic.
    else:
        results = [None, effect_size]  # p-value of the F-statistic.
    # get p-value, t-value, estimate, standard error for each factor
    for factor in factor_names:
        results.extend([fitted_model.pvalues[factor], fitted_model.tvalues[factor], fitted_model.params[factor], fitted_model.bse[factor]] )
    return results

@unique
class DM_TYPE(Enum):
    """Enum for the different types of DM objects (Differentially Methylated Probes or Regions)"""
    DMP = 'DMP'
    DMR = 'DMR'

class DM:

    def __init__(self, samples: Samples, formula: str, reference_value: dict | None = None,
                 custom_sheet: None | pd.DataFrame = None, drop_na=False, apply_mask=True,
                 probe_ids: None | list[str] = None, group_column: str | None = None):
        """Initialize the object by calcating the Differentially Methylated Probes (DMP). It fits an Ordinary Least
         Square model (OLS) for each probe, following the given formula.
         If a group column name is given, use a Mixed Model to account for random effects.

        More info on  design matrices and formulas:
            - https://www.statsmodels.org/devel/gettingstarted.html
            - https://patsy.readthedocs.io/en/latest/overview.html

        :param samples: samples to use
        :type samples: Samples
        :param formula: R-like formula used in the design matrix to describe the statistical model. e.g. '~age + sex'
        :type formula: str
        :param reference_value: reference value for each factor. Dictionary where keys are the factor names, and values are
            their reference value. Default: None
        :type reference_value: dict | None
        :param custom_sheet: a sample sheet to use. By default, use the samples' sheet. Useful if you want to filter the
            samples to display
        :type custom_sheet: pandas.DataFrame
        :param drop_na: drop probes that have NA values. Default: False
        :type drop_na: bool
        :param apply_mask: set to True to apply mask. Default: True
        :type apply_mask: bool
        :param probe_ids: list of probe IDs to use. Useful to work on a subset for testing purposes. Default: None
        :type probe_ids: list[str] | None
        :param group_column: name of the column of the sample sheet that holds replicates information. If provided,
            a Mixed Model will be used to account for replicates instead of an Ordinary Least Square. Default: None
        :type group_column: str | None

        :return: dataframe with probes as rows and p_vales and model estimates in columns, list of contrast levels
        :rtype: pandas.DataFrame, list[str]
        """

        self.dmp = None
        self.dmr = None
        self.contrasts = None
        self.formula = None
        self.samples = None
        self.sample_info = None
        self.group_column = None
        self.reference_value = None
        self.seg_per_locus = None
        self.dist_cutoff = None
        self.segments = None

        self.compute_dmp(samples, formula, reference_value, custom_sheet, drop_na, apply_mask, probe_ids, group_column)

    def get_top(self, dm_type: DM_TYPE | str, contrast:str, chromosome_col='chromosome', annotation_col: str = 'genes',
                n_dms=10, columns_to_keep: list[str] = None) -> pd.DataFrame | None:
        """Get the top DMRs from the dataframe returned by get_dmr(), ranked by the p-value of the given contrast. If an
        annotation is provided, the DMRs will be annotated with the genes associated with the probes in the DMR.

        :param dm_type: type of Differentially Methylated object to get (DMR or DMP).
        :type dm_type: DM_TYPE | str

        :param contrast: contrast to use for ranking the DMRs
        :type contrast: str

        :param chromosome_col: name of the column holding the chromosome information. Default: 'chromosome'
        :type chromosome_col: str

        :param annotation_col: name of the column holding the annotation information. Default: 'genes'
        :type annotation_col: str

        :param n_dms: number of DM probes/segments to return. Default: 10
        :type n_dms: int

        :param columns_to_keep: list of columns to keep in the output dataframe. Default: None
        :type columns_to_keep: list[str] | None

        :return: dataframe with the top DMRs
        :rtype: pandas.DataFrame | None
        """
        # check if the input parameters are correct
        if self.samples is None or self.dmp is None or len(self.dmp) == 0:
            LOGGER.error('Please calculate DMP first')
            return

        sort_column =  f'{contrast}_p_value_adjusted'
        columns_to_keep = [] if columns_to_keep is None else columns_to_keep

        if dm_type in [DM_TYPE.DMR, 'DMR']:
            if self.dmr is None or self.segments is None:
                LOGGER.error('Please calculate DMR first')
                return None
            top_dm = self.dmr.join(self.segments.reset_index().set_index('segment_id'))
            if chromosome_col not in top_dm.columns:
                LOGGER.error(f'Chromosome column {chromosome_col} was not found in the dataframe')
                return None
            columns_to_keep += [chromosome_col, 'probe_id']
            top_dm[chromosome_col] = merge_alt_chromosomes(top_dm[chromosome_col])
        else:
            top_dm = self.dmp

        if sort_column not in top_dm.columns:
            LOGGER.error(f'The column {sort_column} for contrast {contrast} was not found in the dataframe. '
                         f'Available contrasts: {self.contrasts}')
            return None

        top_dm = top_dm[[sort_column] + columns_to_keep].dropna(subset=sort_column)

        # check annotation parameter, and select and clean up annotation if defined
        probe_infos = self.samples.annotation.probe_infos
        if annotation_col not in probe_infos.columns:
            LOGGER.warning(f'{annotation_col} was not found in the annotation dataframe. '
                         f'Available columns : {probe_infos.columns}.')
        else:
            gene_info = probe_infos[['probe_id', annotation_col]].drop_duplicates().set_index('probe_id')
            gene_info = gene_info.dropna().reset_index()
            top_dm = top_dm.reset_index().merge(gene_info, how='inner', on='probe_id').drop(columns='probe_id')

            # todo time optimization: filter with the first N probes/segments then do that
            group_columns = top_dm.columns.tolist()
            group_columns.remove(annotation_col)
            top_dm = top_dm.reset_index(drop=True).drop_duplicates().groupby(group_columns).agg(merge_series_values)
            top_dm[annotation_col] = top_dm[annotation_col].apply(lambda x: ';'.join(set(x.split(';'))))

        return top_dm.sort_values(sort_column).iloc[:n_dms]


    def compute_dmp(self, samples: Samples, formula: str, reference_value: dict | None = None,
                 custom_sheet: None | pd.DataFrame = None, drop_na=False, apply_mask=True,
                 probe_ids: None | list[str] = None, group_column: str | None = None):
        """Find Differentially Methylated Probes (DMP) by fitting an Ordinary Least Square model (OLS) for each probe,
        following the given formula. If a group column name is given, use a Mixed Model to account for random effects.

        More info on  design matrices and formulas:
            - https://www.statsmodels.org/devel/gettingstarted.html
            - https://patsy.readthedocs.io/en/latest/overview.html

        :param samples: samples to use
        :type samples: Samples
        :param formula: R-like formula used in the design matrix to describe the statistical model. e.g. '~age + sex'
        :type formula: str
        :param reference_value: reference value for each factor. Dictionary where keys are the factor names, and values are
            their reference value. Default: None
        :type reference_value: dict | None
        :param custom_sheet: a sample sheet to use. By default, use the samples' sheet. Useful if you want to filter the samples to display
        :type custom_sheet: pandas.DataFrame
        :param drop_na: drop probes that have NA values. Default: False
        :type drop_na: bool
        :param apply_mask: set to True to apply mask. Default: True
        :type apply_mask: bool
        :param probe_ids: list of probe IDs to use. Useful to work on a subset for testing purposes. Default: None
        :type probe_ids: list[str] | None
        :param group_column: name of the column of the sample sheet that holds replicates information. If provided,
            a Mixed Model will be used to account for replicates instead of an Ordinary Least Square. Default: None
        :type group_column: str | None

        :return: dataframe with probes as rows and p_vales and model estimates in columns, list of contrast levels
        :rtype: pandas.DataFrame, list[str]
        """

        LOGGER.info('>>> Start calculating DMP')
        if custom_sheet is None:
            custom_sheet = samples.sample_sheet.copy()

        # check the sample sheet
        if samples.sample_label_name not in custom_sheet.columns:
            LOGGER.error(f'get_dmp() : the provided sample sheet must have a "{samples.sample_label_name}" column')
            return None, None

        # if a group column is specified, check the input
        if group_column is not None:
            if group_column not in custom_sheet.columns:
                LOGGER.error(f'The group column {group_column} was not found in the sample sheet columns')
                return None, None
            if pd.isna(custom_sheet[group_column]).any():
                LOGGER.warning(f'The group column {group_column} has NA values, dropping the corresponding samples.')
                custom_sheet = custom_sheet[~pd.isna(custom_sheet[group_column])].copy()

        # check factors
        factor_columns = get_factors_from_formula(formula)
        for c in factor_columns:
            if c not in custom_sheet.columns:
                LOGGER.error(f'The factor {c} was not found in the sample sheet columns')
                return None, None
            if pd.isna(custom_sheet[c]).any():
                LOGGER.warning(f'NA values where found in the {c} column of the sample sheet. The corresponding samples will be dropped')
                custom_sheet = custom_sheet[~pd.isna(custom_sheet[c])].copy()

        betas = samples.get_betas(drop_na=drop_na, apply_mask=apply_mask, custom_sheet=custom_sheet)

        if betas is None:
            LOGGER.error('No probes left')
            return None, None

        # drop probes with only NAs even if drop_na is false
        if not drop_na:
            betas = betas.dropna(how='all')
        if len(betas) == 0:
            LOGGER.error('No probes left')
            return None, None

        betas = set_level_as_index(betas, 'probe_id', drop_others=True)
        if probe_ids is not None:
            probe_ids = betas.index.intersection(probe_ids)
            betas = betas.loc[probe_ids]
        # make the design matrix
        sample_info = custom_sheet[custom_sheet[samples.sample_label_name].isin(betas.columns)]
        sample_info = sample_info.set_index(samples.sample_label_name)
        # order betas and sample_info the same way
        sample_names_order = [c for c in betas.columns if c in sample_info.index]
        sample_info = sample_info.loc[sample_names_order]
        betas = betas[sample_names_order]
        groups_info = sample_info[group_column] if group_column is not None else None

        # the reference level for each factor is the first level of the sorted factor values. If a specific reference value
        # is provided, we sort the levels accordingly
        if reference_value is not None:
            for column_name, value in reference_value.items():
                if column_name in sample_info.columns:
                    order = [value] + [v for v in set(sample_info[column_name]) if v != value]
                    sample_info[column_name] = pd.Categorical(sample_info[column_name], categories=order, ordered=True)
        try:
            design_matrix = dmatrix(formula, sample_info, return_type='dataframe')
        except:
            design_matrix = pd.DataFrame()

        # check that the design matrix is not empty (it happens for example if the variable used in the formula is constant)
        if len(design_matrix.columns) < 2:
            LOGGER.error('The design matrix is empty. Please make sure the formula you provided is correct.')
            return None, None

        factor_names = [f for f in design_matrix.columns]
        column_names = ['f_pvalue', 'effect_size']
        column_names += [f'{factor}_{c}' for factor in factor_names for c in ['p_value', 't_value', 'estimate', 'std_err']]

        # if it's a small dataset, don't parallelize
        if len(betas) <= 10000:
            result_array = [_get_model_parameters(row[1:], design_matrix, factor_names, groups_info) for row in betas.itertuples()]
        # otherwise parallelize
        else:
            def wrapper_get_model_parameters(row):
                return _get_model_parameters(row, design_matrix, factor_names, groups_info)
            result_array = Parallel(n_jobs=-1)(delayed(wrapper_get_model_parameters)(row[1:]) for row in betas.itertuples())

        dmps = pd.DataFrame(result_array, index=betas.index, columns=column_names, dtype='float64')
        
        LOGGER.info('add average beta delta between groups')

        # get column names used in the formula that are categories or string
        cat_column_names = [c for c in factor_columns if sample_info.dtypes[c] in ['category', 'object']]
        for col in cat_column_names:
            first_factor = None
            for name, group in sample_info.groupby(col, observed=True):
                dmps[f'avg_beta_{col}_{name}'] = betas.loc[:, group.index].mean(axis=1)
                if first_factor is None:
                    first_factor = name
                else:
                    dmps[f'avg_beta_delta_{col}_{first_factor}_vs_{name}'] = dmps[f'avg_beta_{col}_{first_factor}'] - dmps[f'avg_beta_{col}_{name}']

        # adjust p-values
        for f in factor_names:
            non_na_indexes = ~np.isnan(dmps[f'{f}_p_value'])  # any NA in f_p_value column causes BH method to crash
            dmps.loc[non_na_indexes, f'{f}_p_value_adjusted'] = multipletests(dmps.loc[non_na_indexes, f'{f}_p_value'], method='fdr_bh')[1]

        self.dmp = dmps
        self.contrasts = factor_names[1:]
        self.samples = samples
        self.sample_info = custom_sheet
        self.reference_value = reference_value
        self.formula = formula
        self.group_column = group_column

        LOGGER.info('get DMP done')

    def compute_dmr(self, contrast: str | list[str] | None=None, dist_cutoff: float | None = None,
                    seg_per_locus: float = 0.5, probe_ids:None|list[str]=None):
        """Find Differentially Methylated Regions (DMR) based on euclidian distance between beta values

        :param contrast: contrast(s) to use for DMR detection
        :type contrast: str | list[str] | None

        :param dist_cutoff: cutoff used to find change points between DMRs, used on euclidian distance between beta values.
            If set to None (default) will be calculated depending on `seg_per_locus` parameter value. Default: None
        :type dist_cutoff: float | None

        :param seg_per_locus: used if dist_cutoff is not set : defines what quartile should be used as a distance cut-off.
            Higher values leads to more segments. Should be 0 < seg_per_locus < 1. Default: 0.5.
        :type seg_per_locus: float

        :param probe_ids: list of probe IDs to use. Useful to work on a subset for testing purposes. Default: None
        :type probe_ids: list[str] | None

        """

        LOGGER.info('>>> Start get DMR')
        if self.dmr is not None:
            LOGGER.warning('DMR already calculated. Replacing it.')
            self.dmr = None
            self.dist_cutoff = None
            self.seg_per_locus = None
            self.segments = None

        # check if the input parameters are correct
        if self.dmp is None or self.samples is None or self.sample_info is None or self.contrasts is None:
            LOGGER.error('Please calculate DMP first')
            return None

        if isinstance(contrast, str):
            contrast = [contrast]
        if isinstance(contrast, list):
            for c in contrast:
                if c not in self.contrasts:
                    LOGGER.error(f'Contrast {c} not found in DMP list. Please calculate DMP for this contrast first')
                    return None
        if contrast is None:
            contrast = self.contrasts

        if len(contrast) == 0:
            LOGGER.error('No contrast provided')
            return None

        # data initialization
        betas = self.samples.get_betas(drop_na=False, custom_sheet=self.sample_info)
        if betas is None:
            return None

        betas = set_level_as_index(betas, 'probe_id', drop_others=True)

        betas = betas.loc[betas.index.intersection(self.dmp.index)]

        if probe_ids is not None:
            probe_ids = betas.index.intersection(probe_ids).intersection(self.dmp.index)
            betas = betas.loc[probe_ids]

        # get genomic range information (for chromosome id and probe position)
        probe_coords_df = self.samples.annotation.genomic_ranges.drop(columns='strand', errors='ignore')
        non_empty_coords_df = probe_coords_df[probe_coords_df.end > probe_coords_df.start]  # remove 0-width ranges

        betas_no_na = betas.dropna(how='all')  # remove probes with missing values
        cpg_ids = non_empty_coords_df.join(betas_no_na, how='inner')

        # if there was no match, try again after trimming the suffix from the genomic ranges probe IDs
        if len(cpg_ids) == 0:
            non_empty_coords_df.index = non_empty_coords_df.index.map(remove_probe_suffix)
            cpg_ids = non_empty_coords_df.join(betas_no_na)

        if len(cpg_ids) == 0:
            LOGGER.error('No match found between genomic probe coordinates and beta values probe IDs')
            return None

        # sort ranges and identify last probe of each chromosome
        # cpg_ranges = pr.PyRanges(cpg_ids).sort_ranges(natsorting=True)  # to have the same sorting as sesame
        cpg_ranges = pr.PyRanges(cpg_ids.rename(columns={'chromosome':'Chromosome', 'end': 'End', 'start': 'Start',
                                                         'strand': 'Strand'})).sort_ranges()
        next_chromosome = cpg_ranges['Chromosome'].shift(-1)
        last_probe_in_chromosome = cpg_ranges['Chromosome'] != next_chromosome

        # compute Euclidian distance of beta values between two consecutive probes of each sample
        sample_labels = betas.columns
        beta_euclidian_dist = (cpg_ranges[sample_labels].diff(-1) ** 2).sum(axis=1)
        beta_euclidian_dist.iloc[-1] = None  # last probe shouldn't have a distance (default is 0 otherwise)

        # determine cut-off if not provided
        if dist_cutoff is None or dist_cutoff <= 0:
            if not 0 < seg_per_locus < 1:
                LOGGER.warning(f'Invalid parameter `seg_per_locus` {seg_per_locus}, should be in ]0:1[. Setting it to 0.5')
                seg_per_locus = 0.5
            if dist_cutoff is not None and dist_cutoff <= 0:
                LOGGER.warning('Wrong input : euclidian distance cutoff for DMP should be > 0. Recalculating it.')
            dist_cutoff = np.quantile(beta_euclidian_dist.dropna(), 1 - seg_per_locus)  # sesame (keep last probes)
            # dist_cutoff = np.quantile(beta_euclidian_dist[~last_probe_in_chromosome], 1 - seg_per_locus)
            LOGGER.info(f'Segments per locus : {seg_per_locus}')

        LOGGER.info(f'Euclidian distance cutoff for DMP : {dist_cutoff}')

        # find change points
        change_points = last_probe_in_chromosome | (beta_euclidian_dist > dist_cutoff)

        # give a unique ID to each segment
        segment_id = change_points.shift(fill_value=True).cumsum()
        segment_id.name = 'segment_id'

        # merging segments with all probes - including empty ones dropped at the beginning
        segments = probe_coords_df.loc[betas.index].join(segment_id).sort_values('segment_id')

        last_segment_id = segment_id.max()
        LOGGER.info(f'Number of segments : {last_segment_id:,}')

        # assign new segments IDs to NA segments
        # NA segments = betas with NA values or probes with 0-width ranges
        na_segments_indexes = segments.segment_id.isna()
        nb_na_segments = na_segments_indexes.sum()
        if nb_na_segments > 0:
            LOGGER.info(f'Adding {nb_na_segments:,} NA segments')
            segments.loc[na_segments_indexes, 'segment_id'] = [n for n in range(nb_na_segments)] + last_segment_id + 1
            segments.segment_id = segments.segment_id.astype(int)

        # combine probes p-values with segments information
        dmr = segments.join(self.dmp)

        # group segments by ID to compute DMR values
        segments_grouped = dmr.groupby('segment_id')
        seg_dmr = pd.DataFrame()
        seg_dmr['start'] = segments_grouped['start'].min()
        seg_dmr['end'] = segments_grouped['end'].max()
        seg_dmr['chromosome'] = segments_grouped['chromosome'].first()
        # calculate each segment's p-values
        LOGGER.info('combining p-values, it might take a few minutes...')

        for c in contrast:
            pval_col = f'{c}_p_value'
            seg_dmr[pval_col] = segments_grouped[pval_col].apply(combine_p_values_stouffer)

            nb_significant = len(seg_dmr.loc[seg_dmr[pval_col] < 0.05])
            LOGGER.info(f' - {nb_significant:,} significant segments for {c} (p-value < 0.05)')

            # use Benjamini/Hochberg's method to adjust p-values
            idxs = ~np.isnan(seg_dmr[pval_col])  # any NA in segment_p_value column causes BH method to crash
            seg_dmr.loc[idxs, f'{pval_col}_adjusted'] = multipletests(seg_dmr.loc[idxs, pval_col], method='fdr_bh')[1]
            nb_significant = len(seg_dmr.loc[seg_dmr[f'{pval_col}_adjusted'] < 0.05])
            LOGGER.info(f' - {nb_significant:,} significant segments after Benjamini/Hochberg\'s adjustment for {c} (p-value < 0.05)')

        # calculate estimates' means for each factor
        for c in self.dmp.columns:
            if c.endswith('estimate') or c.startswith('avg_beta_'):
                seg_dmr[c] = segments_grouped[c].mean()

        self.segments = segments[['segment_id']]
        self.dmr = seg_dmr