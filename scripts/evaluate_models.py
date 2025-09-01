import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

# Apply global style
sns.set_theme(style="whitegrid")
custom_palette = ["#4E79A7", "#59A14F", "#9C755F", "#F28E2B", "#E15759"]
sns.set_palette(custom_palette)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

def save_plot(fig, filename):
    os.makedirs("images", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    path_images = os.path.join("images", filename)
    path_outputs = os.path.join("outputs/plots", filename)
    fig.savefig(path_images, bbox_inches="tight", dpi=300)
    fig.savefig(path_outputs, bbox_inches="tight", dpi=300)
    plt.close(fig)

def evaluate_and_compare(y_test, y_pred_lr, y_pred_xgb):
    results = {
        "Model": ["Linear Regression", "XGBoost"],
        "R2": [
            r2_score(y_test, y_pred_lr),
            r2_score(y_test, y_pred_xgb)
        ],
        "MSE": [
            mean_squared_error(y_test, y_pred_lr),
            mean_squared_error(y_test, y_pred_xgb)
        ],
        "MAE": [
            mean_absolute_error(y_test, y_pred_lr),
            mean_absolute_error(y_test, y_pred_xgb)
        ]
    }

    df_results = pd.DataFrame(results)

    os.makedirs("reports", exist_ok=True)
    df_results.to_csv("reports/model_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    df_results.set_index("Model")[["MSE", "MAE"]].plot(kind="bar", ax=ax, edgecolor="black")
    ax.set_title("Model Performance Comparison (Errors)", fontsize=14, weight="bold")
    ax.set_ylabel("Error Value")
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_plot(fig, "model_performance_comparison.png")

    print("\n Model Evaluation Results:")
    print(df_results)
    print("\n Results saved to reports/")
    print(" Plots saved to images/ and outputs/plots/")

    return df_results

def generate_extra_plots(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False, ax=ax)
    ax.set_title("Correlation Heatmap", weight="bold")
    save_plot(fig, "correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    df.select_dtypes(include="number").boxplot(ax=ax, rot=45, grid=False)
    ax.set_title("Outlier Boxplots", weight="bold")
    save_plot(fig, "outlier_boxplots.png")

    if "Education_Level" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x="Education_Level", y="Expected_CTC", data=df, ax=ax)
        ax.set_title("Education vs Salary Distribution", weight="bold")
        plt.xticks(rotation=45)
        save_plot(fig, "education_salary_distribution.png")

    if "Designation" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(
            x="Designation",
            data=df,
            ax=ax,
            order=df["Designation"].value_counts().index
        )
        ax.set_title("Designation Distribution", weight="bold")
        plt.xticks(rotation=45)
        save_plot(fig, "designation_distribution.png")

    if "Department" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x="Department", y="Expected_CTC", data=df, ax=ax)
        ax.set_title("Department vs Salary Distribution", weight="bold")
        plt.xticks(rotation=45)
        save_plot(fig, "department_salary_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Expected_CTC"], bins=30, kde=True, ax=ax)
    ax.set_title("Expected Salary Distribution", weight="bold")
    save_plot(fig, "expected_salary_distribution.png")

if __name__ == "__main__":
    print(" Starting model evaluation...")

    df = pd.read_csv("data/processed/df_encoded.csv")
    df = df.fillna(df.mean(numeric_only=True))

    X = df.drop("Expected_CTC", axis=1)
    y = df["Expected_CTC"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr_model = joblib.load("models/linear_regression_model.joblib")
    xgb_model = joblib.load("models/xgboost_model.joblib")

    y_pred_lr = lr_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)

    results_df = evaluate_and_compare(y_test, y_pred_lr, y_pred_xgb)

    print("\n Generating extra plots...")
    generate_extra_plots(df)

    print("\n Evaluation complete.")
