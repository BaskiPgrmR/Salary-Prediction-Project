import pandas as pd
import os
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_group_metrics(df, target_col, pred_col, group_cols):
    """
    Compute fairness metrics (MAE, RMSE) per group.
    """
    records = []
    grouped = df.groupby(group_cols, dropna=False)  # allow missing groups if any
    for name, g in grouped:
        y_true, y_pred = g[target_col], g[pred_col]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        records.append({
            "group": name if isinstance(name, tuple) else (name,),
            "n": len(g),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2)
        })
    return pd.DataFrame(records)


def fairness_summary(groups, overall_mae):
    """
    Compare group MAE vs overall MAE.
    """
    groups["mae_gap"] = groups["mae"] - overall_mae
    return {
        "overall_mae": round(overall_mae, 2),
        "max_gap": round(groups["mae_gap"].abs().max(), 2),
        "mean_gap": round(groups["mae_gap"].abs().mean(), 2),
    }


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv("reports/df_banded.csv")

    overall_mae = mean_absolute_error(df["Expected_CTC"], df["offered_salary"])

    group_cols = [c for c in ["Department", "Role", "Designation", "Education"] if c in df.columns]
    if not group_cols:
        raise ValueError("No valid grouping columns found in dataset!")

    groups = compute_group_metrics(df, "Expected_CTC", "offered_salary", group_cols)
    groups.to_csv("reports/fairness_by_group.csv", index=False)

    summary = fairness_summary(groups, overall_mae)
    with open("reports/fairness.json", "w") as f:
        json.dump({"overall_mae": overall_mae, "fairness": summary}, f, indent=2)

    print("Fairness audit complete. Reports saved in reports/")
