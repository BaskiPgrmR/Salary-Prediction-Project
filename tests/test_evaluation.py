import numpy as np
from scripts.evaluate_models import evaluate_and_compare

def test_evaluation():
    y_test = np.array([100, 200, 300])
    y_pred_lr = np.array([110, 190, 310])
    y_pred_xgb = np.array([105, 205, 295])
    results = evaluate_and_compare(y_test, y_pred_lr, y_pred_xgb)
    assert "R2" in results.columns
