import pandas as pd
import numpy as np
import pytest

from pylluminator.dm import _get_model_parameters, DM


def test_dmp_ols(test_samples):
    probe_ids = test_samples.get_signal_df().reset_index()['probe_id'].sort_values()[:1000].tolist()
    my_dms = DM(test_samples, '~ sample_type', probe_ids=probe_ids)
    assert my_dms.dmp.loc['cg00000029_TC21', 'Intercept_estimate'] == pytest.approx(0.7574986020723979)  # Est_X.Intercept.
    assert my_dms.dmp.loc['cg00000029_TC21', 'Intercept_p_value'] == pytest.approx(4.257315170037784e-06)  # Pval_X.Intercept.
    assert my_dms.dmp.loc['cg00000029_TC21', 'sample_type[T.PREC]_estimate'] == pytest.approx(-0.7096783705055714)  # Est_sample_typeP
    assert my_dms.dmp.loc['cg00000029_TC21', 'sample_type[T.PREC]_p_value'] == pytest.approx(2.1946549071376195e-05)  # Pval_sample_typeP
    assert my_dms.dmp.loc['cg00000029_TC21', 'effect_size'] ==  pytest.approx(0.7096783705055714)  # Eff_sample_type

    my_dms = DM(test_samples, '~ sample_type + sample_number', probe_ids=probe_ids)
    assert my_dms.dmp.loc['cg00000029_TC21', 'Intercept_estimate'] == pytest.approx(0.8015912032375739) # Est_X.Intercept.
    assert my_dms.dmp.loc['cg00000029_TC21', 'Intercept_p_value'] == pytest.approx(0.0003027740974405947)  # Pval_X.Intercept.
    assert my_dms.dmp.loc['cg00000029_TC21', 'sample_type[T.PREC]_estimate'] == pytest.approx(-0.7096783705055711)  # Est_sample_typeP
    assert my_dms.dmp.loc['cg00000029_TC21', 'sample_type[T.PREC]_p_value'] == pytest.approx(0.00015479425278001256) # Pval_sample_typeP
    assert my_dms.dmp.loc['cg00000029_TC21', 'sample_number_estimate'] == pytest.approx(-0.02204630058258784)  # Est_sample_typeP
    assert my_dms.dmp.loc['cg00000029_TC21', 'sample_number_p_value'] == pytest.approx(0.30724222260281375) # Pval_sample_typeP
    assert my_dms.dmp.loc['cg00000029_TC21', 'effect_size'] == pytest.approx(0.7096783705055711)  # Eff_sample_type

    top_10_dmps = my_dms.get_top_dmp('sample_type[T.PREC]')    
    assert len(top_10_dmps) == 10
    assert 'CCDC181' in top_10_dmps.iloc[0].genes
    assert 'ENSG00000285535' in top_10_dmps.iloc[1].genes


def test_dmp_mixedmodel(test_samples, caplog):
    probe_ids = test_samples.get_signal_df().reset_index()['probe_id'].sort_values()[:1000].tolist()
    test_samples.sample_sheet['sentrix_id'] = pd.NA

    # column with Nas
    caplog.clear()
    my_dms = DM(test_samples, '~ sample_type', group_column='sentrix_id', probe_ids=probe_ids)
    assert 'The group column sentrix_id has NA values' in caplog.text
    assert my_dms.dmp is None
    assert my_dms.contrasts is None

    # non existing column
    caplog.clear()
    my_dms = DM(test_samples, '~ sample_type', group_column='notfound', probe_ids=probe_ids)
    assert 'The group column notfound was not found' in caplog.text
    assert my_dms.dmp is None
    assert my_dms.contrasts is None

    # OK
    caplog.clear()
    test_samples.sample_sheet['sentrix_position'] = [name[-1:] for name in test_samples.sample_sheet['sample_name']]
    my_dms = DM(test_samples, '~ sentrix_position', group_column='sentrix_position', probe_ids=probe_ids)
    assert 'ERROR' not in caplog.text
    assert my_dms.dmp is not None
    assert my_dms.contrasts is not None

def test_dmp_bad_sample_sheet(test_samples, caplog):
    # test missing value in factor column
    probe_ids = test_samples.get_signal_df().reset_index()['probe_id'].sort_values()[:1000].tolist()
    test_samples.sample_sheet.loc[3, 'sample_type'] = np.NAN
    caplog.clear()
    my_dms = DM(test_samples, '~ sample_type', probe_ids=probe_ids)
    assert 'NA values where found in the sample_type column of the sample sheet' in caplog.text
    assert my_dms.dmp is not None
    assert my_dms.contrasts is not None

    # test missing value in group column
    probe_ids = test_samples.get_signal_df().reset_index()['probe_id'].sort_values()[:1000].tolist()
    test_samples.sample_sheet.loc[2, 'sample_number'] = np.NAN
    caplog.clear()
    my_dms = DM(test_samples, '~ sample_type', group_column='sample_number', probe_ids=probe_ids)
    assert 'The group column sample_number has NA values' in caplog.text
    assert my_dms.dmp is not None
    assert my_dms.contrasts is not None

    # test missing sample_name column
    test_samples.sample_sheet = test_samples.sample_sheet.drop(columns='sample_name')
    caplog.clear()
    my_dms = DM(test_samples, '~ sample_type')
    assert 'the provided sample sheet must have a "sample_name" column' in caplog.text
    assert my_dms.dmp is None
    assert my_dms.contrasts is None

def test_dmp_wrong_formula(test_samples):
    # non existent factor column
    my_dms = DM(test_samples, '~ nonexistent_factor')
    assert my_dms.dmp is None
    assert my_dms.contrasts is None

    
def test_reference_value(test_samples, caplog):
    # non existent factor column
    caplog.clear()
    my_dms = DM(test_samples, '~ sample_type', reference_value='PREC')
    assert 'parameter reference_value must be a dict' in caplog.text
    assert my_dms.dmp is None
    assert my_dms.contrasts is None

    caplog.clear()
    my_dms = DM(test_samples, '~ sample_type', reference_value={'bad_key': 'PREC'})
    assert 'predictor bad_key was not found in sample metadata' in caplog.text
    assert my_dms.dmp is None
    assert my_dms.contrasts is None
    
    caplog.clear()
    my_dms = DM(test_samples, '~ sample_type', reference_value={'sample_type': 'PREC'})
    assert my_dms.dmp is not None
    assert my_dms.contrasts is not None


def test_ols_na():
    nb_factors = 3
    params = _get_model_parameters([np.nan] * 5, pd.DataFrame(), ['factor'] * nb_factors)
    assert len(params) == 2 + nb_factors * 4
    assert np.isnan(params).all()

def test_dmr(test_samples):
    probe_ids = test_samples.get_signal_df().reset_index()['probe_id'].sort_values()[:1000].tolist()
    # probe_ids.extend(['cg14515812_TC11', 'cg14515812_TC12'])
    my_dms = DM(test_samples, '~ sample_type', probe_ids=probe_ids)

    my_dms.compute_dmr()

    # check segments
    assert max(my_dms.segments.segment_id) == 516
    assert len(my_dms.segments[my_dms.segments.segment_id == 515]) == 3
    assert my_dms.segments.loc['cg00017004_BC21', 'segment_id'] == 515

    # check DMRs values
    expected_values = [151960303, 153792416, 'X', 0.04285787432065091, 0.06373101772177485, 0.7505345278316073,
                       0.055821167098151304, -0.05582112]
    assert my_dms.dmr.loc[515, ].values.tolist() == pytest.approx(expected_values)


def test_get_top(test_samples, caplog):
    probe_ids = test_samples.get_signal_df().reset_index()['probe_id'].sort_values()[:1000].tolist()
    my_dms = DM(test_samples, '~ sample_type', probe_ids=probe_ids)

    # test get_top function before compute DMRs = bug
    caplog.clear()
    my_dms.get_top_dmr('sample_type[T.PREC]')
    assert 'Please calculate DMRs first' in caplog.text

    my_dms.compute_dmr()

    # OK
    caplog.clear()
    top_10_dmrs = my_dms.get_top_dmr('sample_type[T.PREC]')
    assert len(top_10_dmrs) == 10
    assert 'CCDC181' in top_10_dmrs.iloc[0].genes
    assert 'ERROR' not in caplog.text
    assert 'WARNING' not in caplog.text

    # NOK
    caplog.clear()
    top_10_dmrs = my_dms.get_top_dmr('sample_type[T.PREC]', chromosome_col='unknown')
    assert 'Chromosome column unknown was not found in the dataframe'
    assert top_10_dmrs is None

    caplog.clear()
    top_10_dmrs = my_dms.get_top_dmr(contrast='unknown')
    assert 'Column unknown_p_value_adjusted for contrast unknown wasn\'t found in'
    assert top_10_dmrs is None
    
    caplog.clear()
    top_10_dmrs = my_dms.get_top_dmr('sample_type[T.PREC]', annotation_col='unknown')
    assert 'annotation_col was not found in the annotation dataframe.'
    assert top_10_dmrs is None

    # more than 2 contrasts
    my_dms = DM(test_samples, '~ sample_number + sample_type', probe_ids=probe_ids)
    my_dms.compute_dmr()

    caplog.clear()
    top_10_dmrs = my_dms.get_top_dmr()
    assert 'More than one contrast available'
    assert top_10_dmrs is None

def test_select_dmp(test_samples):
    my_dms = DM(test_samples, '~ sample_type')
    selected_dmps = my_dms.select_dmps()
    assert len(selected_dmps) == 937688

    sort_col = 'sample_type[T.PREC]_p_value_adjusted'
    selected_dmps = my_dms.select_dmps(p_value_th=0.01, effect_size_th=0.2, sort_by=sort_col, ascending=True)
    assert len(selected_dmps) == 307093
    assert selected_dmps.iloc[0].name == 'cg17049328_TC21'
    assert selected_dmps.iloc[0][sort_col] == min(selected_dmps[sort_col])