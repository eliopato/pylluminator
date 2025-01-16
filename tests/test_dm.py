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
    return samples

def test_dmp(test_samples):
    indexes_py = test_samples.get_signal_df().reset_index()['probe_id'].sort_values()[:1000].tolist()
    dmps, contrasts = get_dmp(test_samples, '~ sample_type', probe_ids=indexes_py)
    assert dmps['cg00000029_TC21', 'Intercept_estimate'] == 0.757499
    assert dmps['cg00000029_TC21', 'sample_type[T.PREC]_estimate'] == -0.709678
    assert dmps['cg00000029_TC21', 'Intercept_p_value'] == 4.257315e-06
    assert dmps['cg00000029_TC21', 'sample_type[T.PREC]_p_value'] == 0.000022
    assert dmps['cg00000029_TC21', 'f_pvalue'] == 0.000022
    assert dmps['cg00000029_TC21', 'effect_size'] == 0.710

