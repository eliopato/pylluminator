import pytest
import os
from pylluminator.samples import read_samples
from pylluminator.utils import download_from_geo

@pytest.fixture(scope='session')
def data_path():
    return os.path.expanduser('~/data/pylluminator-utestgi')

@pytest.fixture(scope='session')
def test_samples_ini(data_path):
    geo_ids = ['GSM7698438', 'GSM7698446', 'GSM7698462', 'GSM7698435', 'GSM7698443', 'GSM7698459']
    download_from_geo(geo_ids, data_path)

    min_beads = 0
    samples = read_samples(data_path, annotation=None, min_beads=min_beads)
    samples.sample_sheet['sample_type'] = [n.split('_')[0] for n in samples.sample_sheet.sample_name]
    samples.sample_sheet['sample_number'] = [int(n[-1]) for n in samples.sample_sheet.sample_name]
    return samples

@pytest.fixture
def test_samples(data_path, test_samples_ini):
    return test_samples_ini.copy()