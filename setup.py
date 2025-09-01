from setuptools import setup, find_packages

setup(
    name="salary-prediction-ml",
    version="1.0.0",
    description="Salary Prediction using ML (Linear Regression & XGBoost)",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "matplotlib",
        "seaborn",
        "joblib"
    ],
)
