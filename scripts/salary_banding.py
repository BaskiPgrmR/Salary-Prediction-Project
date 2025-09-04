import pandas as pd
import os
import json


def apply_salary_bands(df: pd.DataFrame, bands: pd.DataFrame, group_cols=None) -> pd.DataFrame:
    if group_cols is None:
        group_cols = []

    valid_cols = [c for c in group_cols if c in df.columns and c in bands.columns]

    if {"band_min", "band_max"}.issubset(bands.columns):
        lower_col, upper_col = "band_min", "band_max"
    elif {"lower", "upper"}.issubset(bands.columns):
        bands = bands.rename(columns={"lower": "band_min", "upper": "band_max"})
        lower_col, upper_col = "band_min", "band_max"
    else:
        raise ValueError("Bands DataFrame must contain ['lower','upper'] or ['band_min','band_max'].")

    if valid_cols:
        merged = df.merge(bands, on=valid_cols, how="left")
    else:
        merged = df.assign(
            band_min=bands[lower_col].iloc[0],
            band_max=bands[upper_col].iloc[0]
        )

    merged["offered_salary"] = merged["predicted_salary"].clip(
        merged[lower_col], merged[upper_col]
    )
    merged["band_clamped"] = (merged["offered_salary"] != merged["predicted_salary"]).astype(int)

    return merged


if __name__ == "__main__":
    raw_path = "data/raw/expected_ctc.csv"
    preds_path = "reports/df_with_predictions.csv"
    bands_path = "data/reference/salary_bands.csv"

    raw = pd.read_csv(raw_path)
    preds = pd.read_csv(preds_path)

    if len(raw) != len(preds):
        raise ValueError("Mismatch: raw dataset and predictions file have different row counts")
    if "predicted_salary" not in preds.columns:
        raise ValueError("Predictions file must contain 'predicted_salary' column")

    df = raw.copy()
    df["predicted_salary"] = preds["predicted_salary"]

    group_cols = [c for c in ["Role", "Department", "Education", "Designation", "Education_Level"] if c in df.columns]

    if group_cols:
        bands = df.groupby(group_cols)["Expected_CTC"].agg(["mean", "std"]).reset_index()
    else:
        bands = df[["Expected_CTC"]].agg(["mean", "std"]).T.reset_index(drop=True)

    bands["band_min"] = (bands["mean"] - 0.5 * bands["std"]).clip(lower=0)
    bands["band_max"] = bands["mean"] + 0.5 * bands["std"]

    os.makedirs("data/reference", exist_ok=True)
    bands.to_csv(bands_path, index=False)
    print(f"[INFO] Salary bands saved to {bands_path}")

    df_banded = apply_salary_bands(df, bands, group_cols)

    os.makedirs("reports", exist_ok=True)
    df_banded.to_csv("reports/df_banded.csv", index=False)

    report = {
        "total": len(df_banded),
        "clamped": int(df_banded["band_clamped"].sum()),
        "grouping_columns": group_cols,
    }
    with open("reports/band_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Salary banding complete. Reports saved in reports/")
