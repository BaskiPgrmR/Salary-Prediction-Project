import pandas as pd
from scripts.salary_banding import apply_salary_bands

def test_salary_banding():
    preds = pd.DataFrame({
        "Role":["Analyst","Manager"],
        "Department":["HR","Engineering"],
        "Education":["Grad","PG"],
        "predicted_salary":[300000,2000000]
    })
    bands = pd.DataFrame({
        "Role":["Analyst","Manager"],
        "Department":["HR","Engineering"],
        "Education":["Grad","PG"],
        "band_min":[400000,700000],
        "band_max":[700000,1400000]
    })
    out = apply_salary_bands(preds, bands)
    assert "offered_salary" in out.columns
    assert all(out["offered_salary"].between(out["band_min"], out["band_max"], inclusive="both"))
