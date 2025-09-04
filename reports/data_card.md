# Data Card: Salary Prediction Dataset

## Dataset
- File: `data/raw/expected_ctc.csv`
- Rows: 25,000
- Columns: 29 (including Role, Department, Designation, Education, Expected_CTC)

## Preprocessing
- Missing values handled
- Outliers treated
- One-hot encoding applied → `data/processed/df_encoded.csv`

## Target
- `Expected_CTC`: Expected salary (INR)

## Known Issues
- Self-reported CTC may be biased
- Sensitive demographics not included (reduces bias risk but limits analysis)
