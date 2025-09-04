# Model Card: Salary Prediction

## Overview
This model predicts fair salary offers based on candidate attributes, while enforcing
role/department/education-based salary ranges to reduce bias.

## Data
- Source: `data/raw/expected_ctc.csv`
- Rows: ~25k
- Features: Role, Department, Designation, Education, Experience, etc.
- Target: Expected_CTC (annual INR)

## Model
- Algorithms: Linear Regression, XGBoost
- Final chosen: XGBoost (best R²)

## Performance
- MAE: ~24,000 INR
- R²: ~0.999
- Metrics stored in `outputs/reports/model_results.csv`

## Fairness
- Groups checked: Department, Role, Designation, Education
- Reports: `fairness.json`, `fairness_by_group.csv`

## Limitations
- Possible proxy bias (education, department)
- Assumes input data is clean and representative
