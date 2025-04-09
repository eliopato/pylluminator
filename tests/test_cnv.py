import pytest

from pylluminator.annotations import Annotations, ArrayType, GenomeVersion
from pylluminator.cnv import copy_number_segmentation, get_normalization_samples, copy_number_variation


def test_norm_samples():
    norm_samples_ev2 = get_normalization_samples(Annotations(ArrayType.HUMAN_EPIC_V2, GenomeVersion.HG38))
    assert norm_samples_ev2 is not None
    assert norm_samples_ev2.nb_samples == 2

    norm_samples_e = get_normalization_samples(Annotations(ArrayType.HUMAN_EPIC, GenomeVersion.HG38))
    assert norm_samples_e is not None
    assert norm_samples_e.nb_samples == 5

    assert get_normalization_samples(Annotations(ArrayType.MOUSE_MM285, GenomeVersion.MM39)) is None

# Copy number variation tests

def test_cnv_default(test_samples):
    all_cnv = copy_number_variation(test_samples)
    assert all_cnv is not None
    assert len(all_cnv.columns) == 10
    assert len(all_cnv) == 937690

def test_cnv_wrong_sample_name(test_samples):
    cnv_df = copy_number_variation(test_samples, sample_labels='erferf')
    assert cnv_df is None

def test_cnv_wrong_normalization_sample_name(test_samples):
    cnv_df = copy_number_variation(test_samples, sample_labels='PREC_500_3', normalization_labels='kopko')
    assert cnv_df is None

def test_cnv_wrong_inputs(test_samples):

    assert copy_number_variation(test_samples, sample_labels='PREC_500_3', group_by='kopko') is None

    assert copy_number_variation(test_samples, group_by='kopko') is None

    test_samples.annotation = Annotations(ArrayType.HUMAN_MSA, GenomeVersion.HG38)  # no default normalization samples
    assert copy_number_variation(test_samples, 'PREC_500_3') is None


def test_cnv_groupby(test_samples):
    cnv_df = copy_number_variation(test_samples, group_by='sample_type')
    assert cnv_df is not None
    assert 'cnv_LNCAP' in cnv_df.columns
    assert 'cnv_PREC' in cnv_df.columns

    cnv_df2 = copy_number_variation(test_samples, group_by='sample_type', normalization_labels='LNCAP')
    assert cnv_df2 is not None
    assert 'cnv_PREC' in cnv_df2.columns


# Copy number segmentation tests

def test_cns_default(test_samples):
    cnv_df = copy_number_variation(test_samples, sample_labels='PREC_500_3')
    ranges, signal_bins_df, segments_df = copy_number_segmentation(test_samples, cnv_df, 'PREC_500_3')
    assert ranges is not None
    assert signal_bins_df is not None
    assert segments_df is not None
    # hard to really test the values as there is randomness in the results
    chr14 = segments_df[segments_df.chromosome == '14']
    assert chr14.values[0].tolist() == pytest.approx(['14', 19187179, 106866859, 726, -0.012314], rel=1e-4)

def test_cns_control(test_samples):
    normalization_samples = ['LNCAP_500_1', 'LNCAP_500_2', 'LNCAP_500_3']
    cnv_df = copy_number_variation(test_samples, sample_labels='PREC_500_3', normalization_labels=normalization_samples)
    ranges, signal_bins_df, segments_df = copy_number_segmentation(test_samples, cnv_df, 'cnv_PREC_500_3')
    assert ranges is not None
    assert signal_bins_df is not None
    assert segments_df is not None
    # hard to really test the values as there is randomness in the results
    chr3 = segments_df[segments_df.chromosome == '3']
    assert chr3.values[0].tolist() == pytest.approx(['3', 180000, 198092780, 1320, -0.091685], rel=1e-4)

def test_cns_single_control(test_samples):
    cnv_df = copy_number_variation(test_samples, sample_labels='PREC_500_3', normalization_labels='LNCAP_500_2')
    ranges, signal_bins_df, segments_df = copy_number_segmentation(test_samples, cnv_df, 'cnv_PREC_500_3')
    assert ranges is not None
    assert signal_bins_df is not None
    assert segments_df is not None

def test_cns_wrong_sample_name(test_samples):
    cnv_df = copy_number_variation(test_samples, sample_labels='PREC_500_3', normalization_labels='LNCAP_500_2')
    assert copy_number_segmentation(test_samples, cnv_df, 'wrongname') == (None, None, None)
