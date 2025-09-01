import pandas as pd
from sklearn.model_selection import train_test_split
from scripts.train_linear_regression import train_linear_regression

def test_lr_training():
    df = pd.read_csv("data/processed/df_encoded.csv")
    X = df.drop("Expected_CTC", axis=1)
    y = df["Expected_CTC"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_linear_regression(X_train, y_train, "models/test_lr.joblib")
    assert model is not None
