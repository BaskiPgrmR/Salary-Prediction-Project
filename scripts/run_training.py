import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import joblib

from scripts.data_preprocessing import load_and_clean_data
from scripts.train_linear_regression import train_linear_regression
from scripts.train_xgboost import train_xgboost
from scripts.evaluate_models import evaluate_and_compare

# Paths
raw_data = "data/raw/expected_ctc.csv"
cleaned_path = "data/processed/df_cleaned.csv"
encoded_path = "data/processed/df_encoded.csv"

# Step 1: Load + preprocess
df, df_encoded = load_and_clean_data(raw_data, cleaned_path, encoded_path)

X = df_encoded.drop("Expected_CTC", axis=1)
y = df_encoded["Expected_CTC"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train models
lr_model = train_linear_regression(X_train, y_train, "models/linear_regression_model.joblib")
xgb_model = train_xgboost(X_train, y_train, "models/xgb_model.joblib")

# Step 3: Evaluate
y_pred_lr = lr_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

results = evaluate_and_compare(y_test, y_pred_lr, y_pred_xgb)
print("Evaluation Results:\n", results)
