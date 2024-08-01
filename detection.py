import pandas as pd
import numpy as np
import logging
from statsmodels.distributions.empirical_distribution import ECDF as ecdf

from utils import get_column_as_flat_array
from sample import Sample

LOGGER = logging.getLogger(__name__)


def pOOBAH(sample: Sample, use_negative_controls=True, threshold=0.05) -> pd.DataFrame:
    """Detection P-value based on empirical cumulative distribution function (ECDF) of out-of-band signal
    aka pOOBAH (p-vals by Out-Of-Band Array Hybridization).
    Parameter `threshold` is used to output a mask based on the p_values.
    Return a dataframe with columns `p_value` and `mask`."""

    # mask non-unique probes - but first save previous mask to reset it afterward
    previous_unmasked_indexes = sample.indexes_not_masked
    sample.add_mask(sample.annotation.non_unique_mask_names)

    # Background = out-of-band type 1 probes + (optionally) negative controls
    background_df = sample.get_oob()
    if use_negative_controls:
        neg_controls = sample.get_negative_controls()
        background_df = pd.concat([background_df, neg_controls])

    bg_green = get_column_as_flat_array(background_df, 'G', remove_na=True)
    bg_red = get_column_as_flat_array(background_df, 'R', remove_na=True)

    if np.sum(bg_red, where=~np.isnan(bg_red)) <= 100:
        LOGGER.info('Not enough out of band signal, use empirical prior')
        bg_red = [n for n in range(1000)]

    if np.sum(bg_green, where=~np.isnan(bg_green)) <= 100:
        LOGGER.info('Not enough out of band signal, use empirical prior')
        bg_green = [n for n in range(1000)]

    # reset mask
    sample.indexes_not_masked = previous_unmasked_indexes

    pval_green = 1 - ecdf(bg_green)(sample.df[['G']].max(axis=1))
    pval_red = 1 - ecdf(bg_red)(sample.df[['R']].max(axis=1))
    p_values = np.min([pval_green, pval_red], axis=0)
    pval_mask = p_values <= threshold

    return pd.DataFrame(data={'p_value': p_values, 'mask': pval_mask},
                        index=sample.df.index)
