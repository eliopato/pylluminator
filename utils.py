import numpy as np
import pandas as pd


def column_names_to_snake_case(df: pd.DataFrame) -> pd.DataFrame:
    """converts the dataframe's column names from camel case to snake case, and replace spaces by underscores"""
    # regex to detect a new word in a camel case string
    camel_case = '(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[a-zA-Z])(?=[1-9])|(?<=[1-9])(?=[A-Z])'
    # specificity, replace CpG by CPG otherwise it becomes cp_g with the regex
    df.columns = df.columns.str.replace('CpG', 'CPG')
    # convert camel case to snake case
    df.columns = df.columns.str.replace(camel_case, '_', regex=True).str.lower()
    df.columns = [c.replace(' ', '_') for c in df.columns]
    return df


def concatenate_non_na(row: pd.Series, col_names: list[str]) -> list:
    """Function to concatenate values of N columns into a list, excluding NaN"""
    values = []
    for col_name in col_names:
        if pd.notna(row[col_name]):
            values.append(row[col_name])
    return values


def get_column_as_flat_array(df: pd.DataFrame, column: str | list, remove_na: bool = False):
    values = df[[column]].values
    if remove_na:
        return values[~np.isnan(values)]
    return values.flatten()
