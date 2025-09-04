import pandas as pd
import os

RAW_DATA_PATH = "data/raw/expected_ctc.csv"
PROCESSED_DIR = "data/processed"
CLEANED_PATH = os.path.join(PROCESSED_DIR, "df_cleaned.csv")
ENCODED_PATH = os.path.join(PROCESSED_DIR, "df_encoded.csv")

def load_and_clean_data(input_path, output_path_cleaned, output_path_encoded):
    df = pd.read_csv(input_path)

    cat_fill = {
        "Department": "Unknown",
        "Role": "Unknown",
        "Industry": "Unknown",
        "Organization": "Unknown",
        "Designation": "Unknown",
        "Education": "Unknown",
        "Graduation_Specialization": "Unknown",
        "University_Grad": "Unknown",
        "PG_Specialization": "Unknown",
        "University_PG": "Unknown",
        "PHD_Specialization": "None",
        "University_PHD": "None",
        "Curent_Location": "Unknown",
        "Preferred_location": "Unknown"
    }
    for col, fill_val in cat_fill.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_val)

    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    df.to_csv(output_path_cleaned, index=False)

    df_encoded = pd.get_dummies(df, drop_first=True)
    df_encoded.to_csv(output_path_encoded, index=False)

    return df, df_encoded


if __name__ == "__main__":
    print("🚀 Starting preprocessing...")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df_cleaned, df_encoded = load_and_clean_data(
        RAW_DATA_PATH,
        CLEANED_PATH,
        ENCODED_PATH
    )

    print(f" Cleaned dataset saved → {CLEANED_PATH} ({df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} cols)")
    print(f" Encoded dataset saved → {ENCODED_PATH} ({df_encoded.shape[0]} rows, {df_encoded.shape[1]} cols)")
    print(" Preprocessing complete!")
