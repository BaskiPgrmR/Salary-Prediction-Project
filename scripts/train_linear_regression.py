import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

def train_linear_regression(X_train, y_train, save_path):
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    return model

if __name__ == "__main__":
    print(" Starting Linear Regression training...")

   
    df = pd.read_csv("data/processed/df_encoded.csv")

    df = df.fillna(df.mean(numeric_only=True))

    
    X = df.drop("Expected_CTC", axis=1)
    y = df["Expected_CTC"]

 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

   
    os.makedirs("models", exist_ok=True)

  
    save_path = "models/linear_regression_model.joblib"
    model = train_linear_regression(X_train, y_train, save_path)

    print(f" Linear Regression model trained and saved → {save_path}")
