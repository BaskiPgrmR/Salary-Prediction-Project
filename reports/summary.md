summary_text = """
Final Report: Salary Prediction Project

1. Introduction
The purpose of this project is to predict the expected salary (CTC) of candidates using historical employee data.
The dataset includes candidate profiles such as education, experience, designation, and department.
The target variable is Expected_CTC.

2. Data Preprocessing
- Missing values handled as follows:
  - Numeric columns: replaced with the median.
  - Categorical columns: replaced with the mode.
- One-hot encoding was applied to categorical columns to make them suitable for machine learning models.

3. Correlation Analysis
- Strongest positive correlation: Expected CTC vs Current CTC (0.99)
- Other positive correlations:
  - Total Experience (0.72)
  - Field Experience (0.68)
- Negative correlations:
  - Graduation Year (-0.68)
  - PhD Year (-0.63)

Visualization: See the correlation heatmap in outputs/plots/correlation_heatmap.png

4. Model Performance
The dataset was modeled using Linear Regression and XGBoost.

Model Performance Table:
-------------------------------------------------
| Model              | R²     | MSE         | MAE     |
|--------------------|--------|-------------|---------|
| Linear Regression  | 0.9960 | 5,379,000,000 | 50,421 |
| XGBoost            | 0.9989 | 1,486,000,000 | 24,302 |
-------------------------------------------------

XGBoost significantly outperformed Linear Regression in all performance metrics.
Visualization: See reports/plots/model_performance_comparison.png

5. Salary Banding & Clamping
- Predicted salaries were clamped to computed salary bands to ensure fairness and consistency.
- Clamping ensures that extreme predicted values are adjusted within acceptable salary ranges.

Key Metrics:
- Number of salaries clamped: df_banded["band_clamped"].sum()
- Group-wise clamping rate: reports/plots/clamping_rate_Department.png

Visualizations:
1. Predicted vs Offered Salaries: reports/plots/salary_band_scatter.png
2. Department-wise Boxplot: reports/plots/band_boxplot_Department.png
3. Role-wise Boxplot: reports/plots/band_boxplot_Role.png
4. Clamping Rate per Group: reports/plots/clamping_rate_Department.png, reports/plots/clamping_rate_Role.png

These plots show where adjustments occurred and highlight salary distribution across roles and departments.

6. Fairness & Insights
- Salary banding reduces manual bias and ensures equity among candidates with similar profiles.
- Higher education levels tend to receive higher median salaries, but clamping ensures fairness within the computed bands.
- Certain departments or roles have more clamped salaries, indicating that the model corrected extreme predictions.

Visualization examples:
- Boxplots per Education Level, Department, and Role
- Histograms of offered vs predicted salary: reports/plots/offered_salary_distribution.png

7. Conclusion
- The goal of this project was to build a robust, fair, and transparent salary prediction system.
- XGBoost was identified as the most accurate model.
- Applying salary bands ensures fairness, reduces manual adjustments, and provides HR teams with actionable insights.
- The project enables consistent salary benchmarking across employees with similar profiles.

8. How to Interpret the Plots
- Correlation Heatmap: Darker colors indicate stronger correlations; shows which factors influence salary most.
- Model Performance Comparison: Lower MSE/MAE and higher R² indicate better performance.
- Predicted vs Offered Salary Scatter: Points on the diagonal are unchanged; red points indicate clamped salaries.
- Boxplots by Department/Role/Education: The boxes show the salary range; points outside the boxes show variability.
- Clamping Rate Plots: Higher bars indicate more salaries were adjusted within that group.
- Histogram of Offered Salaries: Shows overall salary distribution after applying bands.
"""

