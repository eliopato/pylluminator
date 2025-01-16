import pytest
import os
import pandas as pd
import numpy as np

from pylluminator.samples import read_samples

@pytest.fixture
def test_samples():
    min_beads = 0
    max_samples = 2
    data_path = os.path.expanduser('~/data/pylluminator-utest')
    return read_samples(data_path, annotation=None, min_beads=min_beads, max_samples=max_samples, keep_idat=True)

def test_poobah(test_samples):
    test_samples.poobah('PREC_500_3')
    test_df = test_samples.get_signal_df()
    assert isinstance(test_df[('PREC_500_3', 'p_value')], pd.Series)
    poobah = test_df[('PREC_500_3', 'p_value')]
    assert sum(np.isnan(poobah)) == 46259
    assert test_samples.masks.number_probes_masked(sample_name='PREC_500_3') == 46259
    assert test_samples.masks.number_probes_masked('poobah_0.05') == 0
    assert test_samples.masks.number_probes_masked('poobah_0.05', 'PREC_500_3') == 46212

def test_quality_mask(test_samples):
    test_samples.apply_quality_mask()
    assert test_samples.masks.number_probes_masked(sample_name='PREC_500_3') == 32948

def test_infer_infinium_I_channel(test_samples):
    summary = test_samples.infer_type1_channel('PREC_500_3')
    assert (summary == [44984, 52, 701, 82558]).all()
    # comparison with R - one probe is different (cg09773691_BC11) because it has < 0 beads in a channel and is set to NA
    # in pylluminator, while in R the other channel values are kept
    # df_r = pd.read_csv('~/diff_r.csv', index_col='Probe_ID')
    # test_samples.infer_type1_channel('PREC_500_3')
    # df_py = test_samples['PREC_500_3'].reset_index().set_index('probe_id')
    # dfs_after = df_r.join(df_py.droplevel('methylation_state', axis=1))
    # dfs_after[dfs_after.col != dfs_after.channel]

def test_dye_bias_linear(test_samples):
    test_samples.dye_bias_correction('PREC_500_3')
    # should be 3424,625, 67547,79, 898.522, 2944.005
    expected_values = [288.32562255859375, 5686.97412109375, 154.75692749023438, 507.06072998046875]
    assert (test_samples.get_probes('cg00002033_TC12')['PREC_500_3'].values == expected_values).all() # Type I green
    expected_values = [213.75865173339844, 205.05917358398438, 1456.2840576171875, 1399.2308349609375]
    assert (test_samples.get_probes('rs6991394_BC11')['PREC_500_3'].values == expected_values).all() # Type I red
    expected_values = [3054.76025390625, 2941.80810546875]  # values 1 and 2 are NA
    assert (test_samples.get_probes('rs9363764_BC21')['PREC_500_3'].values[0, [0, 3]] == expected_values).all() # Type II
