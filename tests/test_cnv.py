import pytest

from pylluminator.cnv import copy_number_variation


def test_cnv_default(test_samples):
    ranges, signal_bins_df, segments_df = copy_number_variation(test_samples, sample_name='PREC_500_3')
    chr14 = segments_df[segments_df.chromosome == '14']
    assert chr14.values[0].tolist() == pytest.approx(['14', 19187179, 106866859, 726, -0.012314287014305592], rel=1e-4)

def test_cnv_control(test_samples):
    normalization_samples = ['LNCAP_500_1', 'LNCAP_500_2', 'LNCAP_500_3']
    ranges, signal_bins_df, segments_df = copy_number_variation(test_samples, sample_name='PREC_500_3',
                                                                normalization_samples_names=normalization_samples)
    chr3 = segments_df[segments_df.chromosome == '3']
    assert chr3.values[0].tolist() == pytest.approx(['3', 180000, 198092780, 1320, -0.09257211536169052], rel=1e-4)
