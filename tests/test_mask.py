from pylluminator.mask import Mask, MaskCollection
import pytest

def test_wrong_input():
    with pytest.raises(ValueError):
        Mask('test', 'test', 'test')
    with pytest.raises(ValueError):
        MaskCollection().add_mask('test')

def test_reset_masks(test_samples):
    test_samples.masks.reset_masks()
    assert len(test_samples.masks.masks) == 0

def test_reset_masks_2(test_samples):
    test_samples.masks.remove_masks()
    assert len(test_samples.masks.masks) == 0

def test_no_mask(test_samples):
    test_samples.apply_mask_by_names('')
    assert test_samples.masks.number_probes_masked() == 0

    test_samples.apply_mask_by_names([''])
    assert test_samples.masks.number_probes_masked() == 0

    test_samples.apply_mask_by_names([])
    assert test_samples.masks.number_probes_masked() == 0

    test_samples.apply_mask_by_names(None)
    assert test_samples.masks.number_probes_masked() == 0

    test_samples.apply_mask_by_names(['uniq'])
    assert test_samples.masks.number_probes_masked() == 23664


def test_remove_sample_mask(test_samples):
    assert test_samples.masks.number_probes_masked(sample_label='LNCAP_500_3') == 26
    assert test_samples.masks.number_probes_masked(sample_label='PREC_500_2') == 21
    test_samples.masks.remove_masks(sample_label='LNCAP_500_3')
    assert test_samples.masks.number_probes_masked(sample_label='LNCAP_500_3') == 0
    assert test_samples.masks.number_probes_masked(sample_label='PREC_500_2') == 21
    assert len(test_samples.masks.masks) == 5

def test_remove_specific_mask(test_samples):
    assert test_samples.masks.number_probes_masked('min_beads_1', 'LNCAP_500_3') == 26
    assert test_samples.masks.number_probes_masked('min_beads_1', 'PREC_500_2') == 21
    test_samples.masks.remove_masks('min_beads_1', 'LNCAP_500_3')
    assert test_samples.masks.number_probes_masked('min_beads_1', 'LNCAP_500_3') == 0
    assert test_samples.masks.number_probes_masked('min_beads_1', 'PREC_500_2') == 21
    assert len(test_samples.masks.masks) == 5

def test_remove_mask(test_samples):
    assert test_samples.masks.number_probes_masked(sample_label='LNCAP_500_3') == 26
    test_samples.masks.remove_masks(mask_name='min_beads_1')
    assert len(test_samples.masks.masks) == 0

def test_get_mask(test_samples):
    assert test_samples.masks[10] is None
    assert test_samples.masks['PREC_500_1'] is None
    assert test_samples.masks['min_beads_1'] is None
    assert test_samples.masks.get_mask('min_beads_1', 'PREC_500_1') is not None

