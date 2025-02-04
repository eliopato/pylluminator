import pytest

from pylluminator.annotations import Annotations, ArrayType, GenomeVersion
from pylluminator.cnv import copy_number_variation, get_normalization_samples

def test_norm_samples():
    norm_samples_ev2 = get_normalization_samples(Annotations(ArrayType.HUMAN_EPIC_V2, GenomeVersion.HG38))
    assert norm_samples_ev2 is not None
    assert norm_samples_ev2.nb_samples == 2

    norm_samples_e = get_normalization_samples(Annotations(ArrayType.HUMAN_EPIC, GenomeVersion.HG38))
    assert norm_samples_e is not None
    assert norm_samples_e.nb_samples == 5

    assert get_normalization_samples(Annotations(ArrayType.MOUSE_MM285, GenomeVersion.MM39)) is None

def test_cnv_default(test_samples):
    ranges, signal_bins_df, segments_df = copy_number_variation(test_samples, sample_label='PREC_500_3')
    assert ranges is not None
    assert signal_bins_df is not None
    assert segments_df is not None
    # hard to really test the values are there is randomness in the results
    chr14 = segments_df[segments_df.chromosome == '14']
    assert chr14.values[0].tolist() == pytest.approx(['14', 19187179, 106866859, 726, -0.012314], rel=1e-4)

def test_cnv_control(test_samples):
    normalization_samples = ['LNCAP_500_1', 'LNCAP_500_2', 'LNCAP_500_3']
    ranges, signal_bins_df, segments_df = copy_number_variation(test_samples, sample_label='PREC_500_3',
                                                                normalization_samples_ids=normalization_samples)
    assert ranges is not None
    assert signal_bins_df is not None
    assert segments_df is not None
    # hard to really test the values are there is randomness in the results
    chr3 = segments_df[segments_df.chromosome == '3']
    assert chr3.values[0].tolist() == pytest.approx(['3', 180000, 198092780, 1320, -0.091685], rel=1e-4)

def test_cnv_single_control(test_samples):
    ranges, signal_bins_df, segments_df = copy_number_variation(test_samples, sample_label='PREC_500_3',
                                                                normalization_samples_ids='LNCAP_500_2')
    assert ranges is not None
    assert signal_bins_df is not None
    assert segments_df is not None

def test_cnv_wrong_sample_name(test_samples):
    assert copy_number_variation(test_samples, 'wrongname') is None

def test_cnv_wrong_normalization_sample_name(test_samples):
    assert copy_number_variation(test_samples, 'PREC_500_3', 'wrongname') is None

def test_cnv_non_existent_normalization(test_samples):
    test_samples.annotation = Annotations(ArrayType.HUMAN_MSA, GenomeVersion.HG38)
    assert copy_number_variation(test_samples, 'PREC_500_3') is None