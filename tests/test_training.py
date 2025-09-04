import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from joblib import dump

def train_linear_regression(X_train, y_train, model_path: str):
    """
    Train a Linear Regression model and save it.
    Ensures no NaNs remain in the training data.
    """
    if X_train.isnull().any().any():
        X_train = X_train.fillna(0)  
    if y_train.isnull().any():
        y_train = y_train.fillna(y_train.median())

    model = LinearRegression()
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    dump(model, model_path)
    print(f"Linear Regression model saved at {model_path}")
    return model


if __name__ == "__main__":

    df = pd.read_csv("data/processed/df_encoded.csv")
    X = df.drop("Expected_CTC", axis=1)
    y = df["Expected_CTC"]

    model = train_linear_regression(X, y, "models/linear_regression.joblib")

