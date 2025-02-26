import pandas as pd
import os
from pylluminator.sample_sheet import read_from_file

def test_wrong_extension():
    assert read_from_file('tests/data/sample_sheet.txt') is None

def test_no_extension():
    assert read_from_file('tests/data/sample_sheet') is None

def test_wrong_filepath():
    assert read_from_file('tests/data/unkwown.csv') is None

def test_empty_file():
    test_df = pd.DataFrame()
    test_df.to_csv('empty.csv')
    assert read_from_file('empty.csv') is None
    os.remove('empty.csv')

def test_header_only_file():
    test_df = pd.DataFrame(columns=['sample_id'])
    test_df.to_csv('empty.csv')
    assert read_from_file('empty.csv') is None
    os.remove('empty.csv')

# check that reading a sample sheet from file creates the same output as creating it from idats
def test_read_sample_sheet(data_path, test_samples):
    sheet_path = data_path + '/samplesheet.csv'
    read_sheet = read_from_file(sheet_path)
    os.remove(sheet_path)
    assert read_sheet.equals(test_samples.sample_sheet.drop(columns=['sample_type', 'sample_number']))
