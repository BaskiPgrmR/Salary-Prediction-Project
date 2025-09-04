import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate_and_compare(y_test, y_pred_lr, y_pred_xgb):
    """
    Compare Linear Regression vs XGBoost predictions on the same test set.
    Returns a DataFrame with evaluation metrics.
    """

    metrics = {
        "Model": ["LinearRegression", "XGBoost"],
        "RMSE": [
            mean_squared_error(y_test, y_pred_lr, squared=False),
            mean_squared_error(y_test, y_pred_xgb, squared=False),
        ],
        "MAE": [
            mean_absolute_error(y_test, y_pred_lr),
            mean_absolute_error(y_test, y_pred_xgb),
        ],
        "R2": [
            r2_score(y_test, y_pred_lr),
            r2_score(y_test, y_pred_xgb),
        ],
    }

    results = pd.DataFrame(metrics)
    print("Model evaluation complete")
    print(results)

    return results


if __name__ == "__main__":
    import numpy as np
    y_test = np.array([100, 200, 300])
    y_pred_lr = np.array([110, 190, 310])
    y_pred_xgb = np.array([105, 205, 295])
    evaluate_and_compare(y_test, y_pred_lr, y_pred_xgb)
