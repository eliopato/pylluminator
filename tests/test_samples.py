from pylluminator.samples import Samples
import os

def test_no_samples():
    samples = Samples()
    assert samples[0] is None

# just checking that the functions don't crash
def test_print_samples(test_samples):
    test_samples.__str__()
    test_samples.__repr__()

def test_get_item(test_samples):
    # access via sample index
    first_name = test_samples.sample_labels[0]
    assert len(test_samples[0].columns) == 4
    assert test_samples[0].columns[0] == (first_name, 'G', 'M')
    # access via non existent sample index
    assert test_samples[10] is None
    # access via sample name
    assert len(test_samples['LNCAP_500_1'].columns) == 4
    assert test_samples['LNCAP_500_1'].columns[0] == ('LNCAP_500_1', 'G', 'M')
    # access via non existent sample name
    assert test_samples['no sample'] is None
    # access via list of sample names
    assert len(test_samples[['LNCAP_500_1', 'LNCAP_500_2']].columns) == 8
    assert test_samples[['LNCAP_500_1', 'LNCAP_500_2']].columns[0] == ('LNCAP_500_1', 'G', 'M')
    # access via list of sample names with non existing name
    assert len(test_samples[['no sample', 'LNCAP_500_2']].columns) == 4
    assert test_samples[['no sample', 'LNCAP_500_2']].columns[0] == ('LNCAP_500_2', 'G', 'M')

def test_get_probe_ids(test_samples):
    assert len(test_samples.probe_ids) == 937688

def test_get_non_existent_probe_type(test_samples):
    assert len(test_samples.get_probes_with_probe_type('non_existent_probe_type')) == 0

def test_get_no_probes(test_samples):
    assert len(test_samples.get_probes(None)) == 0
    assert len(test_samples.get_probes([])) == 0
    assert len(test_samples.get_probes('non existent id')) == 0
    assert len(test_samples.get_probes(['fake probe', 'non existent id'])) == 0

def test_save_load(test_samples):
    test_samples.save('test_samples')
    test_samples2 = Samples.load('test_samples')
    os.remove('test_samples')
    assert test_samples.nb_samples == test_samples2.nb_samples
    assert test_samples.nb_probes == test_samples2.nb_probes
    assert test_samples.sample_sheet.equals(test_samples2.sample_sheet)
    assert test_samples.masks.number_probes_masked() == test_samples2.masks.number_probes_masked()

def test_load_nonexistent():
    test_object = Samples.load('nonexistent_file')
    assert test_object is None

def test_get_sigdf(test_samples):
    assert test_samples.get_signal_df(apply_mask=True).equals(test_samples.get_signal_df(apply_mask='wrong value'))

def test_merge_samples(test_samples):
    test_samples.merge_samples_by('patient_id')
    assert test_samples.sample_label_name == 'sample_name'  # it should not have changed

    test_samples.merge_samples_by('sample_name')
    assert test_samples.sample_label_name == 'sample_name'  # it should not have changed

    test_samples.calculate_betas()
    test_samples.merge_samples_by('sample_type')
    assert test_samples.sample_label_name == 'sample_type'
    assert test_samples.masks.number_probes_masked(sample_label='LNCAP') == 0
    assert test_samples.masks.number_probes_masked(sample_label='PREC') == 0

def test_remove_probes_suffix(test_samples):
    test_samples.remove_probes_suffix()
    sigdf = test_samples.get_signal_df()
    assert len(sigdf) == 931390
    test_samples.calculate_betas()
    gr_df = test_samples.annotation.genomic_ranges
    assert len(gr_df.loc[sigdf.index.get_level_values('probe_id')[:100]]) == 100

def test_drop_samples(test_samples):
    test_samples.drop_samples('PREC_500_3')
    assert 'PREC_500_3' not in test_samples._signal_df.columns
    assert 'PREC_500_3' not in test_samples._betas.columns
    assert len(test_samples.masks.get_mask_names('PREC_500_3')) == 0
    assert 'PREC_500_3' not in test_samples.sample_sheet[test_samples.sample_label_name].values
    assert 'PREC_500_3' not in test_samples.sample_labels
    test_samples.drop_samples(['PREC_500_3', 'unexistent'])

def test_subset_samples(test_samples):
    test_samples.subset('PREC_500_3')
    assert 'PREC_500_3' in test_samples._signal_df.columns
    assert 'PREC_500_3' in test_samples._betas.columns
    assert 'LNCAP_500_3' not in test_samples._signal_df.columns
    assert 'LNCAP_500_3' not in test_samples._betas.columns
    assert len(test_samples.masks.get_mask_names('PREC_500_3')) > 0
    assert 'PREC_500_3' in test_samples.sample_sheet[test_samples.sample_label_name].values
    assert 'PREC_500_3' in test_samples.sample_labels
    assert 'LNCAP_500_3' not in test_samples.sample_labels
    assert 'LNCAP_500_3' not in test_samples.sample_sheet[test_samples.sample_label_name].values
    test_samples.subset(['PREC_500_3', 'unexistent'])