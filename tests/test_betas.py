import pandas as pd
from pytest import approx
from pylluminator.annotations import ArrayType

# test beta values on object my_samples
def test_calculate_betas(test_samples):
    test_samples.calculate_betas(include_out_of_band=False)
    betas = test_samples.get_betas()['PREC_500_3']
    assert betas.xs('cg00002033_TC12', level="probe_id").to_numpy() == 0.04825291  # Type I green
    assert betas.xs('rs6991394_BC11', level="probe_id").to_numpy() == 0.50999004  # Type I red
    assert betas.xs('rs9363764_BC21', level="probe_id").to_numpy() == 0.373386 # Type II

    test_samples.calculate_betas(include_out_of_band=True)
    betas = test_samples.get_betas()['PREC_500_3']
    assert betas.xs('cg00002033_TC12', level="probe_id").to_numpy() == 0.07827754  # Type I green
    assert betas.xs('rs6991394_BC11', level="probe_id").to_numpy() ==  0.51002073  # Type I red
    assert betas.xs('rs9363764_BC21', level="probe_id").to_numpy() == 0.373386 # Type II

def test_betas_options(test_samples):
    # test sample_label and custom_sheet options
    assert test_samples.get_betas(sample_label='unkwown') is None

    # test sample_label option
    test_df = test_samples.get_betas(sample_label='PREC_500_3')
    assert len(test_df) == 937688
    assert isinstance(test_df, pd.Series)
    assert test_df.iloc[0] == test_samples.get_betas()['PREC_500_3'].iloc[0]

    # test sample_label and custom_sheet options (a warning should be triggered for using both sample_label and custom_sheet)
    test_df = test_samples.get_betas(sample_label='PREC_500_3', custom_sheet=pd.DataFrame())
    assert len(test_df) == 937688
    assert isinstance(test_df, pd.Series)
    assert test_df.iloc[0] == test_samples.get_betas()['PREC_500_3'].iloc[0]

    # test custom_sheet option
    custom_sheet = test_samples.sample_sheet[test_samples.sample_sheet[test_samples.sample_label_name] == 'LNCAP_500_3']
    test_df = test_samples.get_betas(custom_sheet=custom_sheet)
    assert len(test_df) == 937688
    assert isinstance(test_df, pd.DataFrame)
    assert len(test_df.columns) == 1
    assert test_df['LNCAP_500_3'].iloc[0] == test_samples.get_betas()['LNCAP_500_3'].iloc[0]

    # test missing sample sheet column
    assert test_samples.get_betas(custom_sheet=custom_sheet.drop(columns=test_samples.sample_label_name)) is None

    # test no samples name matching beta columns
    custom_sheet.loc[:, test_samples.sample_label_name] = custom_sheet['sample_type']
    assert test_samples.get_betas(custom_sheet=custom_sheet) is None

def test_liftover_impute(test_samples, caplog):
    
    # check the values before doing anything
    test_samples.calculate_betas()
    assert pd.isna(test_samples._betas[['LNCAP_500_1']].xs(key='cg02909018_BC11', level='probe_id').values[0][0])
    assert test_samples._betas[['LNCAP_500_2']].xs(key='cg02909018_BC11', level='probe_id').values[0][0] == 0.8948475

    # TEST LIFT OVER

    # lift over, check the signal df and the betas
    test_samples.lift_over_probe_annotations(ArrayType.HUMAN_450K)
    assert len(test_samples._signal_df) == 486427
    assert (test_samples._signal_df[['PREC_500_3']].xs(key='cg00005543', level='probe_id').values == [113, 4886, 182, 1336]).all()
    assert test_samples._betas is None # beta df should be reset
    # beta values should not have changed
    test_samples.calculate_betas()
    assert test_samples._betas[['LNCAP_500_1']].xs(key='cg00003287', level='probe_id').values[0][0] == 0.6121235
    assert pd.isna(test_samples._betas[['LNCAP_500_1']].xs(key='cg02909018', level='probe_id').values[0][0])
    assert test_samples._betas[['LNCAP_500_2']].xs(key='cg02909018', level='probe_id').values[0][0] == 0.8948475

    # TEST IMPUTE BETAS

    # this should autodetect prostate cell type and impute the right betas
    test_samples.impute_betas()
    assert test_samples._betas[['PREC_500_3']].xs(key='cg00005390', level='probe_id').values[0][0] == approx(0.74282684)  # probe that's only in HM450
    assert test_samples._betas[['PREC_500_3']].xs(key='cg00000108', level='probe_id').values[0][0] == approx(0.95204189)
     # check that existing values didnt change
    assert test_samples._betas[['LNCAP_500_1']].xs(key='cg00003287', level='probe_id').values[0][0] == approx(0.6121235)
    assert test_samples._betas[['LNCAP_500_2']].xs(key='cg02909018', level='probe_id').values[0][0] == approx(0.8948475)
    # check the imputation was done correctly
    assert test_samples._betas[['LNCAP_500_1']].xs(key='cg02909018', level='probe_id').values[0][0] == approx(0.97664808)  # this value was missing for this sample only

    # check that the function outputs an error if a wrong cell type is given
    caplog.clear()
    test_samples.impute_betas(celltype='Unknown')
    assert 'Column Unknown.median not found' in caplog.text

    # check that the cell type and sd max are taken into account if provided
    test_samples._betas = None  # reset betas
    test_samples.impute_betas(celltype='Blood', sd_max=0.2)
    assert test_samples._betas[['PREC_500_3']].xs(key='cg00005390', level='probe_id').values[0][0] == approx(0.733508)
    assert test_samples._betas[['PREC_500_3']].xs(key='cg00000108', level='probe_id').values[0][0] == approx(0.968770)
    assert pd.isna(test_samples._betas[['LNCAP_500_1']].xs(key='cg00004073', level='probe_id').values[0][0]) # this one should stay NA as it has a sd above 0.2

