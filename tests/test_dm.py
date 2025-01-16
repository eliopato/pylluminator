import pytest
import os
from pylluminator.dm import get_dmp

from pylluminator.samples import read_samples

@pytest.fixture
def test_samples():
    min_beads = 1
    data_path = os.path.expanduser('~/data/pylluminator-utest')
    samples = read_samples(data_path, annotation=None, min_beads=min_beads)
    samples.sample_sheet['sample_type'] = [n.split('_')[0] for n in samples.sample_sheet.sample_name]
    samples.sample_sheet['sample_number'] = [int(n[-1]) for n in samples.sample_sheet.sample_name]
    return samples

def test_dmp(test_samples):
    indexes_py = test_samples.get_signal_df().reset_index()['probe_id'].sort_values()[:1000].tolist()
    dmps, contrasts = get_dmp(test_samples, '~ sample_type', probe_ids=indexes_py)
    assert dmps.loc['cg00000029_TC21', 'Intercept_estimate'] == 0.7574986020723979  # Est_X.Intercept.
    assert dmps.loc['cg00000029_TC21', 'Intercept_p_value'] == 4.257315170037784e-06  # Pval_X.Intercept.
    assert dmps.loc['cg00000029_TC21', 'sample_type[T.PREC]_estimate'] == -0.7096783705055714  # Est_sample_typeP
    assert dmps.loc['cg00000029_TC21', 'sample_type[T.PREC]_p_value'] == 2.1946549071376195e-05  # Pval_sample_typeP
    assert dmps.loc['cg00000029_TC21', 'effect_size'] ==  0.7096783705055714  # Eff_sample_type

    dmps, contrasts = get_dmp(test_samples, '~ sample_type + sample_number', probe_ids=indexes_py)
    assert dmps.loc['cg00000029_TC21', 'Intercept_estimate'] == 0.8015912032375739 # Est_X.Intercept.
    assert dmps.loc['cg00000029_TC21', 'Intercept_p_value'] == 0.0003027740974405947  # Pval_X.Intercept.
    assert dmps.loc['cg00000029_TC21', 'sample_type[T.PREC]_estimate'] == -0.7096783705055711  # Est_sample_typeP
    assert dmps.loc['cg00000029_TC21', 'sample_type[T.PREC]_p_value'] == 0.00015479425278001256 # Pval_sample_typeP
    assert dmps.loc['cg00000029_TC21', 'sample_number_estimate'] == -0.02204630058258784  # Est_sample_typeP
    assert dmps.loc['cg00000029_TC21', 'sample_number_p_value'] == 0.30724222260281375 # Pval_sample_typeP
    assert dmps.loc['cg00000029_TC21', 'effect_size'] == 0.7096783705055711  # Eff_sample_type