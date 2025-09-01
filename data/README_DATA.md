# Raw Data: Salary Prediction Project

This folder contains the **raw dataset** used in the Salary Prediction Project.  
The data has not been cleaned or preprocessed — it is preserved in its original form for reproducibility.

## Dataset Description
The file `expected_ctc.csv` contains candidate information including education, experience, designation, department, and salary-related details. The primary goal is to use this data to predict the expected salary (`Expected_CTC`) of an employee.

### Key Columns
- **Total Experience**: Total years of work experience.  
- **Field Experience**: Years of experience in the relevant field.  
- **Graduation Year**: Year of graduation (if applicable).  
- **PhD Year**: Year of PhD completion (if applicable).  
- **Designation**: Current designation of the candidate.  
- **Department**: Department in which the candidate works.  
- **Current_CTC**: Current annual salary (in INR).  
- **Expected_CTC**: Expected annual salary (in INR) — **target variable**.

### Value Ranges
- **Department** includes values such as:  
  `HR, Top Management, Banking, Sales, Engineering, Others, Analytics/BI, Education, Marketing, Healthcare, IT-Software, Accounts`.  

- **Designation** includes values such as:  
  `HR, Medical Officer, Director, Marketing Manager, Manager, Product Manager, Consultant, CA, Research Scientist, Sr.Manager, Data Analyst, Assistant Manager, Others, Web Designer, Research Analyst, Software Developer, Network Engineer, Scientist`.  

- **Experience Columns**: Range from `0` years (freshers) up to `40+` years.  
- **Graduation Year**: Spans across several decades, primarily `1980s–2020s`.  
- **CTC Values**: Ranges from a few lakhs (entry-level) to several crores (top management).  

## Notes
- This dataset may contain **missing values** and **outliers**.  
- Categorical features (e.g., Department, Designation) will need **encoding** before modeling.  
- Numerical features (e.g., Experience, CTCs, Years) may require **scaling** or **normalization**.  

## Usage
This dataset is used as the foundation for:
1. **Exploratory Data Analysis (EDA)**  
2. **Preprocessing & Feature Engineering**  
3. **Model Training & Evaluation**  

The raw version should not be modified directly. Instead, cleaned and processed datasets are stored in the `data/processed/` folder.
