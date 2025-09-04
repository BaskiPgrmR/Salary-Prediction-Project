import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

# Styling
sns.set_theme(style="whitegrid", font="monospace")
custom_palette = ["#4E79A7", "#59A14F", "#9C755F", "#F28E2B", "#E15759"]
sns.set_palette(custom_palette)
plt.rcParams.update({"figure.dpi":120, "axes.titlesize":14, "axes.labelsize":12})

# Utility to save plots
def save_plot(fig, filename):
    os.makedirs("images", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    fig.savefig(os.path.join("images", filename), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join("outputs/plots", filename), bbox_inches="tight", dpi=300)
    plt.close(fig)

# Model evaluation
def evaluate_models(y_test, y_pred_lr, y_pred_xgb):
    results = {
        "Model": ["Linear Regression", "XGBoost"],
        "R2": [r2_score(y_test, y_pred_lr), r2_score(y_test, y_pred_xgb)],
        "MSE": [mean_squared_error(y_test, y_pred_lr), mean_squared_error(y_test, y_pred_xgb)],
        "MAE": [mean_absolute_error(y_test, y_pred_lr), mean_absolute_error(y_test, y_pred_xgb)]
    }

    df_results = pd.DataFrame(results)

    os.makedirs("reports", exist_ok=True)
    df_results.to_csv("reports/model_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(8,5))
    df_results.set_index("Model")[["MSE","MAE"]].plot(kind="bar", ax=ax, edgecolor="black")
    ax.set_title("Model Performance Comparison (Errors)")
    ax.set_ylabel("Error Value")
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_plot(fig, "model_performance_comparison.png")

    print("\n Model Evaluation Results:")
    print(df_results)
    print("\n Results saved to reports/")
    print(" Plots saved to images/ and outputs/plots/")

    return df_results

# Base exploratory plots
def generate_base_plots(df):
    # Correlation heatmap
    numeric_cols = df.select_dtypes(include="number").columns
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap", weight="bold")
    save_plot(fig, "correlation_heatmap.png")

    # Outlier boxplots
    fig, ax = plt.subplots(figsize=(14, 6))
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols].boxplot(ax=ax, rot=45, grid=False)  # Rotate labels for readability
    ax.set_title("Outlier Boxplots", weight="bold")
    plt.tight_layout() 
    save_plot(fig, "outlier_boxplots.png")

    # Education vs Expected Salary
    if "Education_Level" in df.columns:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.boxplot(x="Education_Level", y="Expected_CTC", data=df, ax=ax)
        ax.set_title("Education vs Salary Distribution", weight="bold")
        plt.xticks(rotation=45)
        save_plot(fig, "education_salary_distribution.png")

    # Department vs Expected Salary
    if "Department" in df.columns:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.boxplot(x="Department", y="Expected_CTC", data=df, ax=ax)
        ax.set_title("Department vs Salary Distribution", weight="bold")
        plt.xticks(rotation=45)
        save_plot(fig, "department_salary_distribution.png")

    # Histogram of Expected Salary
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(df["Expected_CTC"], bins=50, kde=True, color="#4E79A7", alpha=0.8, ax=ax)
    ax.set_title("Expected Salary Distribution", weight="bold")
    save_plot(fig, "expected_salary_distribution.png")

# Salary banding
def apply_salary_bands(df, group_cols, predicted_salary):
    if group_cols:
        bands = df.groupby(group_cols)["Expected_CTC"].agg(["mean","std"]).reset_index()
        bands["band_min"] = (bands["mean"] - 0.5*bands["std"]).clip(lower=0)
        bands["band_max"] = bands["mean"] + 0.5*bands["std"]
        df_banded = df.merge(bands, on=group_cols, how="left")
        offered_salary = predicted_salary.clip(lower=df_banded["band_min"], upper=df_banded["band_max"])
    else:
        mean_val = df["Expected_CTC"].mean()
        std_val = df["Expected_CTC"].std()
        band_min = max(mean_val - 0.5*std_val, 0)
        band_max = mean_val + 0.5*std_val
        offered_salary = predicted_salary.clip(lower=band_min, upper=band_max)

    band_clamped = (offered_salary != predicted_salary).astype(int)
    df_banded = pd.concat([df, predicted_salary.rename("predicted_salary"), offered_salary.rename("offered_salary"), band_clamped.rename("band_clamped")], axis=1)
    return df_banded

def salary_band_visualizations(df, group_cols):
    # Scatterplot
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(x=df["predicted_salary"], y=df["offered_salary"], hue=df["band_clamped"].map({0:"Within Band",1:"Clamped"}),
                    palette={"Within Band":"#4E79A7","Clamped":"#E15759"}, alpha=0.7, s=50, ax=ax)
    ax.plot([df["predicted_salary"].min(), df["predicted_salary"].max()],
            [df["predicted_salary"].min(), df["predicted_salary"].max()],
            color="gray", linestyle="--")
    ax.set_title("Predicted vs Offered Salary")
    save_plot(fig, "salary_band_scatter.png")

    # Boxplots & clamping rates
    for col in group_cols:
        if col in df.columns:
            fig, ax = plt.subplots(figsize=(10,6))
            sns.boxplot(x=col, y="offered_salary", data=df, color="#4E79A7", showfliers=False, ax=ax)
            sns.stripplot(x=col, y="predicted_salary", data=df, color="#F28E2B",
                          size=4, jitter=True, alpha=0.6, ax=ax)
            ax.set_title(f"Salary Banding by {col}")
            save_plot(fig, f"band_boxplot_{col}.png")

            clamped_rate = df.groupby(col)["band_clamped"].mean().reset_index()
            clamped_rate["band_clamped"] *= 100
            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(x=col, y="band_clamped", data=clamped_rate, palette="muted", ax=ax)
            ax.set_ylabel("Clamping Rate (%)")
            ax.set_title(f"Clamping Rate by {col}")
            save_plot(fig, f"clamping_rate_{col}.png")

    # Histogram of offered salary
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(df_banded["offered_salary"], bins=50, kde=True, color="#4E79A7", alpha=0.8, ax=ax)
    ax.set_title("Offered Salary Distribution", weight="bold")
    ax.set_xlabel("Offered Salary")
    ax.set_ylabel("Count")
    plt.tight_layout()
    save_plot(fig, "offered_salary_distribution.png")


# Experience plots
def experience_plots(df):
    if "Total_Experience" in df.columns and "Total_Experience_in_field_applied" in df.columns:
        fig, ax = plt.subplots(1, 2, figsize=(16,6))
        sns.regplot(x='Total_Experience', y='Expected_CTC', data=df,
                    scatter_kws={'color': 'darkorange', 'alpha':0.6, 's':30, 'edgecolor':'w'},
                    line_kws={'color':'darkblue', 'linewidth':2}, ax=ax[0])
        ax[0].set_title('Total Experience vs Expected Salary', fontsize=16, fontweight='bold', fontname='monospace')
        ax[0].set_xlabel('Total Experience (Years)', fontsize=13, fontname='monospace')
        ax[0].set_ylabel('Expected CTC', fontsize=13, fontname='monospace')

        sns.regplot(x='Total_Experience_in_field_applied', y='Expected_CTC', data=df,
                    scatter_kws={'color':'crimson', 'alpha':0.6, 's':30, 'edgecolor':'w'},
                    line_kws={'color':'seagreen', 'linewidth':2}, ax=ax[1])
        ax[1].set_title('Field Experience vs Expected Salary', fontsize=16, fontweight='bold', fontname='monospace')
        ax[1].set_xlabel('Experience in Applied Field (Years)', fontsize=13, fontname='monospace')
        ax[1].set_ylabel('Expected CTC', fontsize=13, fontname='monospace')

        plt.tight_layout()
        save_plot(fig, "experience_vs_expected_salary.png")

# Main execution
if __name__ == "__main__":
    print(" Starting model evaluation...")

    df = pd.read_csv("data/processed/df_encoded.csv")
    df = df.fillna(df.mean(numeric_only=True))

    X = df.drop("Expected_CTC", axis=1)
    y = df["Expected_CTC"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr_model = joblib.load("models/linear_regression_model.joblib")
    xgb_model = joblib.load("models/xgboost_model.joblib")

    y_pred_lr = lr_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)

    results_df = evaluate_models(y_test, y_pred_lr, y_pred_xgb)
    best_model_name = results_df.loc[results_df["R2"].idxmax(),"Model"]
    best_model = xgb_model if best_model_name=="XGBoost" else lr_model

    predicted_salary = pd.Series(best_model.predict(X), index=df.index)

    # Base plots
    generate_base_plots(df)

    # Salary banding plots
    group_cols = [c for c in ["Role","Department","Education_Level"] if c in df.columns]
    df_banded = apply_salary_bands(df, group_cols, predicted_salary)
    df_banded.to_csv("reports/df_banded.csv", index=False)
    salary_band_visualizations(df_banded, group_cols)

    # Experience plots
    experience_plots(df_banded)

    print("Pipeline complete. All plots saved in images/ and outputs/plots.")
