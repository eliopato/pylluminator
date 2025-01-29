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