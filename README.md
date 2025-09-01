# Salary-Prediction-Project
---

## Introduction
In today’s competitive environment, organizations must ensure fairness and consistency when determining employee salaries. Salary benchmarking using historical data helps reduce human bias, minimize discrimination, and bring transparency to Human Resource (HR) decision-making.

This project develops a machine learning-based salary prediction system that uses employee profiles (education, experience, department, designation, etc.) to predict the expected salary. By leveraging data-driven methods, the system automates salary recommendations and supports HR professionals in making consistent and fair decisions.

---

## Goal and Objective
The objective of this project is to build a model, using historical data, that determines the salary to be offered to an employee. The intention is to minimize manual judgment in the selection process while ensuring employees with similar profiles receive fair compensation.  

The project’s goal and objectives were successfully achieved:
- A predictive model was built and trained on real-world-like historical data.  
- The approach reduces subjective bias and improves transparency.  
- The system is robust, interpretable, and capable of providing accurate salary recommendations.  

---

## Dataset Description
The dataset contains profiles of candidates who applied to Company X.  

**Features include:**
- Education: Graduate, Postgraduate, PhD  
- Total Experience: Overall years of work experience  
- Field Experience: Relevant years of work experience  
- Designation: Job role applied for  
- Department: Department in which the candidate works  
- Current CTC: Current salary of the candidate  

**Target Variable:**
- Expected CTC: Salary expected by the candidate  

---

## Methodology

### 1. Data Preprocessing
- Missing values handled:
  - Numeric values filled with the median  
  - Categorical values filled with the mode  
- One-hot encoding applied to categorical variables  
- Standardization applied to numerical features  

### 2. Exploratory Data Analysis (EDA)
- Correlation analysis to identify key salary drivers  
- Distribution plots to understand Expected CTC patterns  
- Department-wise and education-level salary comparisons  

### 3. Model Training
Two regression models were implemented and compared:
- Linear Regression  
- XGBoost Regressor  

### 4. Evaluation
Models were evaluated on:
- R² (Coefficient of Determination)  
- MSE (Mean Squared Error)  
- MAE (Mean Absolute Error)  

---

## Results and Findings

### Correlation Insights
- Strongest correlation: Expected CTC with Current CTC (0.99)  
- Positive correlations: Total Experience, Field Experience  
- Negative correlations: Graduation Year (-0.68), PhD Year (-0.63)  

### Model Performance
| Model              | R²       | MSE          | MAE       |
|--------------------|----------|--------------|-----------|
| Linear Regression  | 0.9960   | 6.37e+09     | 50,421    |
| XGBoost Regressor  | 0.9989   | 1.67e+09     | 24,302    |

XGBoost outperformed Linear Regression, achieving higher accuracy and lower error.  

---

## Visualizations
The following plots were created to support analysis:
- Correlation Heatmap – Feature relationships with salary  
- Salary Distribution Plot – Distribution and skewness of Expected CTC  
- Department Salary Barplot – Department-level salary differences  
- Education Level Boxplot – Salary differences by education  
- Model Comparison Plot – Comparing regression models  

Plots are stored in:
- `outputs/plots/`  
- `images/`  

---


