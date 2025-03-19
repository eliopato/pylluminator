import pandas as pd
import numpy as np
import pytest


def test_poobah(test_samples):
    test_samples.poobah('PREC_500_3')
    test_df = test_samples.get_signal_df()
    assert isinstance(test_df[('PREC_500_3', 'p_value')], pd.Series)
    poobah = test_df[('PREC_500_3', 'p_value')]
    assert sum(np.isnan(poobah)) == 46259
    assert test_samples.masks.number_probes_masked(sample_label='PREC_500_3') == 46259
    assert test_samples.masks.number_probes_masked('poobah_0.05') == 0
    assert test_samples.masks.number_probes_masked('poobah_0.05', 'PREC_500_3') == 46212

def test_quality_mask(test_samples):
    test_samples.mask_quality_probes()
    assert test_samples.masks.number_probes_masked(sample_label='PREC_500_3') == 32948

def test_infer_infinium_I_channel(test_samples):
    summary = test_samples.infer_type1_channel('PREC_500_3')
    assert summary.values.tolist() == [44984, 701, 52, 82558]
    assert len(test_samples.type1_red()) == 83259
    assert len(test_samples.type1_green()) == 45036

    # comparison with R - one probe is different (cg09773691_BC11) because it has < 0 beads in a channel and is set to NA
    # in pylluminator, while in R the other channel values are kept
    # df_r = pd.read_csv('~/diff_r.csv', index_col='Probe_ID')
    # test_samples.infer_type1_channel('PREC_500_3')
    # df_py = test_samples['PREC_500_3'].reset_index().set_index('probe_id')
    # dfs_after = df_r.join(df_py.droplevel('methylation_state', axis=1))
    # dfs_after[dfs_after.col != dfs_after.channel]

    test_samples.infer_type1_channel()

def test_infer_infinium_I_channel_switch(test_samples):
    summary = test_samples.infer_type1_channel('PREC_500_3', switch_failed=True, mask_failed=True)
    assert summary.values.tolist() == [44803, 882, 67, 82543]

def test_dye_bias_corr(test_samples):
    test_samples.dye_bias_correction('PREC_500_3')
    # Type I green
    expected_values = [288.32562255859375, 5686.97412109375, 154.75692749023438, 507.06072998046875]
    assert test_samples.get_probes('cg00002033_TC12')['PREC_500_3'].values[0].tolist() == pytest.approx(expected_values)
    # Type I red
    expected_values = [213.75865173339844, 205.05917358398438, 1456.2840576171875, 1399.2308349609375]
    assert test_samples.get_probes('rs6991394_BC11')['PREC_500_3'].values[0].tolist() == pytest.approx(expected_values)
    # Type II - values 1 and 2 are NA
    expected_values = [3054.76025390625, 2941.80810546875]
    assert test_samples.get_probes('rs9363764_BC21')['PREC_500_3'].values[0, [0, 3]] == pytest.approx(expected_values)

def test_dye_bias_corr_non_existent(test_samples):
    # no change should be done if a wrong sample is passed
    test_samples2 = test_samples.copy()
    test_samples2.dye_bias_correction('no sample')  # non existent sample
    assert test_samples.get_signal_df().equals(test_samples2.get_signal_df())
    test_samples2.dye_bias_correction(2587)  # wrong input type
    assert test_samples.get_signal_df().equals(test_samples2.get_signal_df())

def test_dye_bias_corr_all(test_samples):
    test_samples.dye_bias_correction()
    # Type I green
    expected_values = [288.32562255859375, 5686.97412109375, 154.75692749023438, 507.06072998046875]
    assert test_samples.get_probes('cg00002033_TC12')['PREC_500_3'].values[0].tolist()  == pytest.approx(expected_values)
    # Type I red
    expected_values = [213.75865173339844, 205.05917358398438, 1456.2840576171875, 1399.2308349609375]
    assert test_samples.get_probes('rs6991394_BC11')['PREC_500_3'].values[0].tolist()  == pytest.approx(expected_values)
    # Type II - values 1 and 2 are NA
    expected_values = [3054.76025390625, 2941.80810546875]
    assert test_samples.get_probes('rs9363764_BC21')['PREC_500_3'].values[0, [0, 3]] == pytest.approx(expected_values)

def test_dye_bias_linear(test_samples):
    test_samples.dye_bias_correction_l('PREC_500_3')
    # Type I green
    expected_values = [405.728759765625, 8002.650390625, 205.05152893066406, 671.8508911132812]
    assert test_samples.get_probes('cg00002033_TC12')['PREC_500_3'].values[0].tolist()  == pytest.approx(expected_values)
    # Type I red
    expected_values = [300.7989196777344, 288.5570983886719, 1929.563232421875, 1853.96826171875]
    assert test_samples.get_probes('rs6991394_BC11')['PREC_500_3'].values[0].tolist()  == pytest.approx(expected_values)
    # Type II - values 1 and 2 are NA
    expected_values = [4298.62646484375, 3897.869140625]
    assert test_samples.get_probes('rs9363764_BC21')['PREC_500_3'].values[0, [0, 3]] == pytest.approx(expected_values)

def test_dye_bias_linear_non_existent(test_samples):
    # no change should be done if a wrong sample is passed
    test_samples2 = test_samples.copy()
    test_samples2.dye_bias_correction_l('no sample')  # non existent sample
    assert test_samples.get_signal_df().equals(test_samples2.get_signal_df())
    test_samples2.dye_bias_correction_l(2587)  # wrong input type
    assert test_samples.get_signal_df().equals(test_samples2.get_signal_df())

def test_dye_bias_linear_all(test_samples):
    test_samples.dye_bias_correction_l()
    # Type I green
    expected_values = [405.728759765625, 8002.650390625, 205.05152893066406, 671.8508911132812]
    assert test_samples.get_probes('cg00002033_TC12')['PREC_500_3'].values[0].tolist()  == pytest.approx(expected_values)
    # Type I red
    expected_values = [300.7989196777344, 288.5570983886719, 1929.563232421875, 1853.96826171875]
    assert test_samples.get_probes('rs6991394_BC11')['PREC_500_3'].values[0].tolist()  == pytest.approx(expected_values)
    # Type II - values 1 and 2 are NA
    expected_values = [4298.62646484375, 3897.869140625]
    assert test_samples.get_probes('rs9363764_BC21')['PREC_500_3'].values[0, [0, 3]] == pytest.approx(expected_values)

def test_dye_bias_nonlinear(test_samples):
    # small differences with R as the probes with no signal (0) are not masked properly in Sesame
    test_samples.dye_bias_correction_nl('PREC_500_3')
    # Type I green
    expected_values = [462.5, 6534.5, 166.0, 475.0]
    assert test_samples.get_probes('cg00002033_TC12')['PREC_500_3'].values[0].tolist()  == pytest.approx(expected_values)
    # Type I red
    expected_values = [307.0, 288.0, 1518.5, 1449.0]
    assert test_samples.get_probes('rs6991394_BC11')['PREC_500_3'].values[0].tolist()  == pytest.approx(expected_values)
    # Type II - values 1 and 2 are NA
    expected_values = [3371.5, 3246.0]
    assert test_samples.get_probes('rs9363764_BC21')['PREC_500_3'].values[0, [0, 3]] == pytest.approx(expected_values)

def test_dye_bias_nonlinear_all(test_samples):
    # small differences with R as the probes with no signal (0) are not masked properly in Sesame
    test_samples.dye_bias_correction_nl()
    # Type I green
    expected_values = [462.5, 6534.5, 166.0, 475.0]
    assert test_samples.get_probes('cg00002033_TC12')['PREC_500_3'].values[0].tolist()  == pytest.approx(expected_values)
    # Type I red
    expected_values = [307.0, 288.0, 1518.5, 1449.0]
    assert test_samples.get_probes('rs6991394_BC11')['PREC_500_3'].values[0].tolist()  == pytest.approx(expected_values)
    # Type II - values 1 and 2 are NA
    expected_values = [3371.5, 3246.0]
    assert test_samples.get_probes('rs9363764_BC21')['PREC_500_3'].values[0, [0, 3]] == pytest.approx(expected_values)

def test_dye_bias_correction_nonlinear_distortion(test_samples):
    test_samples._signal_df.loc['I', 'G'] = 0
    test_samples.dye_bias_correction_nl()

def test_noob(test_samples):
    test_samples.noob_background_correction('PREC_500_3')
    # Type I green
    expected_values = [98.03031921386719, 4391.55712890625, 160.515625, 327.8741455078125]
    assert test_samples.get_probes('cg00002033_TC12')['PREC_500_3'].values[0].tolist()  == pytest.approx(expected_values)
    # Type I red
    expected_values = [75.70853424072266, 73.65463256835938, 1540.5257568359375, 1460.5260009765625]
    assert test_samples.get_probes('rs6991394_BC11')['PREC_500_3'].values[0].tolist()  == pytest.approx(expected_values)
    # Type II - values 1 and 2 are NA
    expected_values = [2273.55712890625, 3623.52587890625]
    assert test_samples.get_probes('rs9363764_BC21')['PREC_500_3'].values[0, [0, 3]] == pytest.approx(expected_values)

def test_noob_all(test_samples):
    test_samples.noob_background_correction()
    # Type I green
    expected_values = [98.03031921386719, 4391.55712890625, 160.515625, 327.8741455078125]
    assert test_samples.get_probes('cg00002033_TC12')['PREC_500_3'].values[0].tolist()  == pytest.approx(expected_values)
    # Type I red
    expected_values = [75.70853424072266, 73.65463256835938, 1540.5257568359375, 1460.5260009765625]
    assert test_samples.get_probes('rs6991394_BC11')['PREC_500_3'].values[0].tolist()  == pytest.approx(expected_values)
    # Type II - values 1 and 2 are NA
    expected_values = [2273.55712890625, 3623.52587890625]
    assert test_samples.get_probes('rs9363764_BC21')['PREC_500_3'].values[0, [0, 3]] == pytest.approx(expected_values)

def test_noob_no_mask(test_samples, caplog):
    caplog.clear()
    test_samples.noob_background_correction(apply_mask=False)
    assert 'ERROR' not in caplog.text

def test_scrub(test_samples):
    test_samples.scrub_background_correction('PREC_500_3')
    # Type I green
    expected_values = [58.0, 4402.0, 1.0, 308.0]
    assert test_samples.get_probes('cg00002033_TC12')['PREC_500_3'].values[0].tolist()  == pytest.approx(expected_values)
    # Type I red
    expected_values = [1.0, 1.0, 1639.0, 1559.0]
    assert test_samples.get_probes('rs6991394_BC11')['PREC_500_3'].values[0].tolist()  == pytest.approx(expected_values)
    # Type II - values 1 and 2 are NA
    expected_values =[2284.0, 3722.0]
    assert test_samples.get_probes('rs9363764_BC21')['PREC_500_3'].values[0, [0, 3]] == pytest.approx(expected_values)

def test_scrub_all(test_samples):
    test_samples.scrub_background_correction()
    # Type I green
    expected_values = [58.0, 4402.0, 1.0, 308.0]
    assert test_samples.get_probes('cg00002033_TC12')['PREC_500_3'].values[0].tolist()  == pytest.approx(expected_values)
    # Type I red
    expected_values = [1.0, 1.0, 1639.0, 1559.0]
    assert test_samples.get_probes('rs6991394_BC11')['PREC_500_3'].values[0].tolist()  == pytest.approx(expected_values)
    # Type II - values 1 and 2 are NA
    expected_values =[2284.0, 3722.0]
    assert test_samples.get_probes('rs9363764_BC21')['PREC_500_3'].values[0, [0, 3]] == pytest.approx(expected_values)

def test_mask_xy(test_samples):
    test_samples.mask_xy_probes()
    assert test_samples.masks.number_probes_masked() == 24953

def test_mask_non_unique(test_samples):
    test_samples.mask_non_unique_probes()
    assert test_samples.masks.number_probes_masked() == 23664
    assert test_samples.masks.number_probes_masked(sample_label='PREC_500_3') == 23716
    assert test_samples.masks.number_probes_masked(mask_name='min_beads_1', sample_label='PREC_500_3') == 52

def test_normalization_controls(test_samples):
    norm_controls = test_samples.get_normalization_controls()
    assert len(norm_controls.columns) == 25
    assert len(norm_controls) == 170
    assert len(norm_controls.xs('R', level='channel')) == 85

def test_get_betas_empty_sheet(test_samples):
    df = test_samples.get_betas(custom_sheet=pd.DataFrame())
    assert df is None

def test_get_betas_drop_na(test_samples):
    df = test_samples.get_betas(drop_na=True)
    assert len(df) == 937544

def test_get_betas_no_betas(test_samples, caplog):
    test_samples.reset_betas()
    caplog.clear()
    assert test_samples.get_betas() is None
    assert 'No beta values found' in caplog.text

def test_batch_correction(test_samples, caplog):
    # test column with na values
    test_samples.sample_sheet['sentrix_id'] = pd.NA

    test_samples.batch_correction('sentrix_id', covariates='sample_type')
    assert 'Batch column contains NaN or empty values' in caplog.text
    assert 'ERROR' in caplog.text
    caplog.clear()

    # test wrong number of batch values
    test_samples.batch_correction([1, 2, 1], covariates='sample_type')
    assert 'Batch column length does not match the number of samples' in caplog.text
    assert 'ERROR' in caplog.text
    caplog.clear()

    # everything OK
    test_samples.batch_correction('sample_number', covariates='sample_type')
    assert 'ERROR' not in caplog.text

    # wrong batch column
    test_samples.batch_correction('wrongcolumn')
    assert 'Batch column wrongcolumn not found' in caplog.text

    # wrong covariates column
    test_samples.batch_correction('sample_number', covariates=['wrongcolumn', 'sample_number'])
    assert 'Covariate wrongcolumn not found' in caplog.text
    assert 'Covariate sample_number must be a string' in caplog.text
    assert 'No valid covariates' in caplog.text

    # test without calculated betas
    caplog.clear()
    test_samples.reset_betas()
    test_samples.batch_correction('sentrix_id', covariates='sample_type')
    assert 'No beta values found' in caplog.text
    assert not test_samples.has_betas()
