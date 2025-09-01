import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
import os

def train_xgboost(X_train, y_train, save_path):
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    return model

if __name__ == "__main__":
    print(" Starting XGBoost training...")

    df = pd.read_csv("data/processed/df_encoded.csv")


    df = df.fillna(df.mean(numeric_only=True))

    X = df.drop("Expected_CTC", axis=1)
    y = df["Expected_CTC"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    os.makedirs("models", exist_ok=True)

    save_path = "models/xgboost_model.joblib"
    model = train_xgboost(X_train, y_train, save_path)

    print(f" XGBoost model trained and saved → {save_path}")
