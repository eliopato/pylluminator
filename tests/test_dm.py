import pandas as pd
import numpy as np
from pylluminator.dm import get_dmp, get_dmr, _get_model_parameters


def test_dmp(test_samples):
    probe_ids = test_samples.get_signal_df().reset_index()['probe_id'].sort_values()[:1000].tolist()
    dmps, contrasts = get_dmp(test_samples, '~ sample_type', probe_ids=probe_ids)
    assert dmps.loc['cg00000029_TC21', 'Intercept_estimate'] == 0.7574986020723979  # Est_X.Intercept.
    assert dmps.loc['cg00000029_TC21', 'Intercept_p_value'] == 4.257315170037784e-06  # Pval_X.Intercept.
    assert dmps.loc['cg00000029_TC21', 'sample_type[T.PREC]_estimate'] == -0.7096783705055714  # Est_sample_typeP
    assert dmps.loc['cg00000029_TC21', 'sample_type[T.PREC]_p_value'] == 2.1946549071376195e-05  # Pval_sample_typeP
    assert dmps.loc['cg00000029_TC21', 'effect_size'] ==  0.7096783705055714  # Eff_sample_type

    dmps, contrasts = get_dmp(test_samples, '~ sample_type + sample_number', probe_ids=probe_ids)
    assert dmps.loc['cg00000029_TC21', 'Intercept_estimate'] == 0.8015912032375739 # Est_X.Intercept.
    assert dmps.loc['cg00000029_TC21', 'Intercept_p_value'] == 0.0003027740974405947  # Pval_X.Intercept.
    assert dmps.loc['cg00000029_TC21', 'sample_type[T.PREC]_estimate'] == -0.7096783705055711  # Est_sample_typeP
    assert dmps.loc['cg00000029_TC21', 'sample_type[T.PREC]_p_value'] == 0.00015479425278001256 # Pval_sample_typeP
    assert dmps.loc['cg00000029_TC21', 'sample_number_estimate'] == -0.02204630058258784  # Est_sample_typeP
    assert dmps.loc['cg00000029_TC21', 'sample_number_p_value'] == 0.30724222260281375 # Pval_sample_typeP
    assert dmps.loc['cg00000029_TC21', 'effect_size'] == 0.7096783705055711  # Eff_sample_type


def test_ols_na():
    nb_factors = 3
    params = _get_model_parameters([np.nan] * 5, pd.DataFrame(), ['factor'] * nb_factors)
    assert len(params) == 2 + nb_factors * 4
    assert np.isnan(params).all()


def test_dmr(test_samples):
    probe_ids = test_samples.get_signal_df().reset_index()['probe_id'].sort_values()[:1000].tolist()
    dmps, contrasts = get_dmp(test_samples, '~ sample_type', probe_ids=probe_ids)
    dmrs = get_dmr(test_samples, dmps, contrasts, probe_ids=probe_ids)
    assert max(dmrs.segment_id) == 516
    assert len(dmrs[dmrs.segment_id == 515]) == 3

    expected_values =['X', 152871744, 152871746, 515, 0.06386775794434923, 0.04350292682647744, 2.22147739294547e-07,
                      72.06726215266518, 0.8722137212753298, 0.012102773093109292, 0.06386775794434768,
                      -2.5416666348952917, -0.04350292682647744, 0.01711590585059933, 151960303, 153792416,
                      0.04285787432064722, 0.055877280730961175, 0.7505345278316077, 0.055821167098150985]
    assert dmrs.loc['cg00017004_BC21', ].values.tolist() == expected_values
