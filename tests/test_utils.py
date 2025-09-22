import logging
import numpy as np
import os
import pandas as pd

from pylluminator.utils import (set_logger, get_logger_level, get_logger, remove_probe_suffix, save_object, load_object,
                                merge_alt_chromosomes, get_chromosome_number, set_level_as_index, download_from_link,
                                merge_series_values, merge_dataframe_by)

def test_set_logger():
    set_logger('WARNING')
    assert get_logger_level() == 30

    set_logger(30)
    assert get_logger_level() == 30

    # these should print an error message but not crash
    set_logger(0.6)
    assert get_logger_level() == 30
    set_logger('wrong value')
    assert get_logger_level() == 30
    set_logger(56)
    assert get_logger_level() == 30

    set_logger('DEBUG')
    assert get_logger_level() == 10

def test_get_logger():
    logger = get_logger(20)
    assert logger is not None
    assert isinstance(logger,logging.Logger)
    assert get_logger_level() == 20
    get_logger()
    assert get_logger_level() == 20

def test_remove_probe_suffix():
    test_str = 'cg00002033_TC12'
    assert remove_probe_suffix(test_str) == 'cg00002033'
    test_str = 'cg00002033'
    assert remove_probe_suffix(test_str) == 'cg00002033'
    test_str = ''
    assert remove_probe_suffix(test_str) == ''
    test_str = 'cg00002033_TC12_TC12'
    assert remove_probe_suffix(test_str) == 'cg00002033_TC12'

def test_save_load(test_samples):
    save_object(test_samples, 'test_samples')
    test_samples2 = load_object('test_samples')
    os.remove('test_samples')
    assert test_samples.nb_samples == test_samples2.nb_samples
    assert test_samples.nb_probes == test_samples2.nb_probes
    assert test_samples.sample_sheet.equals(test_samples2.sample_sheet)
    assert test_samples.masks.number_probes_masked() == test_samples2.masks.number_probes_masked()

def test_load_nonexistent():
    test_object = load_object('nonexistent_file')
    assert test_object is None

def test_merge_chromosome_string():
    assert merge_alt_chromosomes(1) == '1'
    assert merge_alt_chromosomes(np.nan) == '*'
    assert merge_alt_chromosomes(None) == '*'
    assert merge_alt_chromosomes('12') == '12'
    assert merge_alt_chromosomes('chr22') == '22'
    assert merge_alt_chromosomes('chrX') == 'x'
    assert merge_alt_chromosomes('chr12_altJI_OIJE') == '12'
    assert merge_alt_chromosomes('chrunk_altJI_OIJE') == '*' # unknown chromosome
    assert merge_alt_chromosomes(['chr88', 'chrM', '12', 90, '*', '22_ALTG66']) == ['88', '*', '12', '90', '*', '22']

def test_get_chromosome_number():
    assert get_chromosome_number(1) == 1
    assert get_chromosome_number(np.nan) is None
    assert get_chromosome_number(None) is None
    assert get_chromosome_number('12') == 12
    assert get_chromosome_number('chr22') == 22
    assert get_chromosome_number('chrX', convert_string=False) is None
    assert get_chromosome_number('chrX', convert_string=True) == 98
    assert get_chromosome_number('chrY', convert_string=True) == 99
    assert get_chromosome_number('chrM', convert_string=True) == 100
    assert get_chromosome_number('chrA', convert_string=True) == 100
    assert get_chromosome_number(['chr88', 'chrM', '12', 90, '*', '22_ALTG66']) == [88, None, 12, 90, None, None]

def test_set_level_as_index(test_samples):
    df = set_level_as_index(test_samples.get_signal_df(), 'probe_id')
    assert ('type', '', '') in df.columns
    assert ('channel', '', '') in df.columns
    assert ('probe_type', '', '') in df.columns
    assert df.index.name == 'probe_id'
    df = set_level_as_index(test_samples.get_signal_df(), 'probe_id', drop_others=True)
    assert ('type', '', '') not in df.columns
    assert ('channel', '', '') not in df.columns
    assert ('probe_type', '', '') not in df.columns
    assert df.index.name == 'probe_id'

def test_failed_download():
    assert download_from_link('https://www.fakeurl.co/fakefile', 'data') == -1


def test_merge_dataframe_by():
    data = pd.DataFrame({
        'A': [1, 1, 2, 2, 3],
        'B': ['x', 'x', 'y', 'y', 'z'],
        'C': [10, 20, 30, 40, 50]
    })
    merged = merge_dataframe_by(data, by=['A', 'B'])
    expected = pd.DataFrame({
        'A': [1, 2, 3], 
        'B': ['x', 'y', 'z'],
        'C': [15.0, 35.0, 50.0]
    }).set_index(['A', 'B'])
    
    pd.testing.assert_frame_equal(merged, expected)

def test_merge_series():
    data = pd.Series([1, 2, 3])
    assert merge_series_values(data) == 2.0

    data = pd.Series([2, 2, 2])
    assert merge_series_values(data) == 2.0

    data = pd.Series(['jo', 'juju', 'bi'])
    assert 'jo' in merge_series_values(data)
    assert 'juju' in merge_series_values(data)
    assert 'bi' in merge_series_values(data)

    data = pd.Series(['jo', 2, 'bi'])
    assert 'jo' in merge_series_values(data)

    data = pd.Series([np.nan, None, np.nan])
    assert merge_series_values(data) is None