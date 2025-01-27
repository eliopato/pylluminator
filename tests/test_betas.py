import pandas as pd


# test beta values on object my_samples
def test_calculate_betas(test_samples):

    betas = test_samples.get_betas(include_out_of_band=False)['PREC_500_3']
    assert betas.xs('cg00002033_TC12', level="probe_id").values == 0.04825291  # Type I green
    assert betas.xs('rs6991394_BC11', level="probe_id").values == 0.50999004  # Type I red
    assert betas.xs('rs9363764_BC21', level="probe_id").values == 0.373386 # Type II

    betas = test_samples.get_betas(include_out_of_band=True)['PREC_500_3']
    assert betas.xs('cg00002033_TC12', level="probe_id").values == 0.07827754  # Type I green
    assert betas.xs('rs6991394_BC11', level="probe_id").values ==  0.51002073  # Type I red
    assert betas.xs('rs9363764_BC21', level="probe_id").values == 0.373386 # Type II

def test_betas_options(test_samples):
    # test sample_name and custom_sheet options
    assert test_samples.get_betas(sample_name="unkwown") is None

    # test sample_name option
    test_df = test_samples.get_betas(sample_name="PREC_500_3")
    assert len(test_df) == 937688
    assert isinstance(test_df, pd.Series)
    assert test_df.iloc[0] == test_samples.get_betas()['PREC_500_3'].iloc[0]

    # test sample_name and custom_sheet options (a warning should be triggered for using both sample_name and custom_sheet)
    test_df = test_samples.get_betas(sample_name="PREC_500_3", custom_sheet=pd.DataFrame())
    assert len(test_df) == 937688
    assert isinstance(test_df, pd.Series)
    assert test_df.iloc[0] == test_samples.get_betas()['PREC_500_3'].iloc[0]

    # test custom_sheet option
    custom_sheet = test_samples.sample_sheet[test_samples.sample_sheet.sample_name == 'LNCAP_500_3']
    test_df = test_samples.get_betas(custom_sheet=custom_sheet)
    assert len(test_df) == 937688
    assert isinstance(test_df, pd.DataFrame)
    assert len(test_df.columns) == 1
    assert test_df['LNCAP_500_3'].iloc[0] == test_samples.get_betas()['LNCAP_500_3'].iloc[0]



# def test_plot_betas(test_samples):
#     plot_betas(test_samples, color_column='sample_group', mask=True)
#     plot_betas(test_samples, color_column='sample_group', mask=False)
#     plot_betas(test_samples, group_column='sample_group', mask=True)
#     plot_betas(test_samples, group_column=['sample_group', 'sample_name_group'], mask=True)
#     plot_betas(test_samples, color_column='sample_group', group_column=['sample_group', 'sample_name_group'])