import pandas as pd
from scripts.data_preprocessing import load_and_clean_data

def test_preprocessing():
    df, df_encoded = load_and_clean_data(
        "data/raw/expected_ctc.csv",
        "data/processed/test_cleaned.csv",
        "data/processed/test_encoded.csv"
    )
    assert not df.isnull().sum().any(), "There are still null values!"
    assert df_encoded.shape[0] == df.shape[0]
