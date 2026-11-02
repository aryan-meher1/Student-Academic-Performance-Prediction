
# ============================================================
# STUDENT ACADEMIC PERFORMANCE PREDICTION
# STEP 1: DATASET INSPECTION AND DATA CLEANING
# ============================================================

import pandas as pd
import os

print("\n==============================================")
print("STUDENT ACADEMIC PERFORMANCE PREDICTION")
print("STEP 1: DATASET INSPECTION AND DATA CLEANING")
print("==============================================")

# ------------------------------------------------------------
# 1. PROJECT FOLDER
# ------------------------------------------------------------

project_folder = r"C:\Users\aryan\Downloads\student performance analysis"

# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

file_path = os.path.join(
    project_folder,
    "Student_Performance_Prediction_Dataset.csv"
)

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")

# ------------------------------------------------------------
# 3. ORIGINAL DATASET INFORMATION
# ------------------------------------------------------------

print("\n========== ORIGINAL DATASET ==========")

print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

# ------------------------------------------------------------
# 4. COLUMN NAMES
# ------------------------------------------------------------

print("\n========== COLUMN NAMES ==========")

print(df.columns.tolist())

# ------------------------------------------------------------
# 5. FIRST FIVE RECORDS
# ------------------------------------------------------------

print("\n========== FIRST FIVE RECORDS ==========")

print(df.head())

# ------------------------------------------------------------
# 6. DATA TYPES
# ------------------------------------------------------------

print("\n========== DATA TYPES ==========")

print(df.dtypes)

# ------------------------------------------------------------
# 7. STANDARDIZE COLUMN NAMES
# ------------------------------------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nColumn names standardized successfully.")

# ------------------------------------------------------------
# 8. HANDLE MISSING VALUE PLACEHOLDERS
# ------------------------------------------------------------

df = df.replace(
    ["?", "NA", "N/A", "NULL", "null", ""],
    pd.NA
)

# ------------------------------------------------------------
# 9. CONVERT NUMERIC COLUMNS
# ------------------------------------------------------------

numeric_columns = [
    "student_id",
    "course_session",
    "previous_attempts",
    "registered_credits",
    "registration_days",
    "previous_marks",
    "assessment_marks",
    "total_assessment_weight",
    "final_assessment_marks",
    "number_of_assessments",
    "assessment_types",
    "active_study_days"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

print("\nNumeric columns converted successfully.")

# ------------------------------------------------------------
# 10. DUPLICATE RECORDS
# ------------------------------------------------------------

print("\n========== DUPLICATE RECORDS ==========")

duplicate_count = df.duplicated().sum()

print("Total Duplicate Records:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicate records removed.")
else:
    print("No duplicate records found.")

# ------------------------------------------------------------
# 11. UNIQUE STUDENTS
# ------------------------------------------------------------

print("\n========== UNIQUE STUDENTS ==========")

print("Unique Students:", df["student_id"].nunique())

# ------------------------------------------------------------
# 12. PASS/FAIL VALIDATION
# ------------------------------------------------------------

print("\n========== PASS/FAIL DISTRIBUTION ==========")

df["pass_fail"] = (
    df["pass_fail"]
    .astype("string")
    .str.strip()
    .str.upper()
)

print(df["pass_fail"].value_counts(dropna=False))

# Keep only valid PASS and FAIL records
df = df[
    df["pass_fail"].isin(["PASS", "FAIL"])
].copy()

print("\nFinal Pass/Fail Distribution:")

print(df["pass_fail"].value_counts())

# ------------------------------------------------------------
# 13. MISSING VALUE ANALYSIS
# ------------------------------------------------------------

print("\n========== MISSING VALUE ANALYSIS ==========")

missing_count = df.isnull().sum()

missing_percentage = df.isnull().mean() * 100

missing_report = pd.DataFrame({
    "Missing_Count": missing_count,
    "Missing_Percentage": missing_percentage.round(2)
})

print(missing_report)

# ------------------------------------------------------------
# 14. CREATE OUTPUT FOLDERS
# ------------------------------------------------------------

outputs_folder = os.path.join(
    project_folder, "outputs"
)

graphs_folder = os.path.join(
    outputs_folder, "graphs"
)

results_folder = os.path.join(
    outputs_folder, "results"
)

models_folder = os.path.join(
    outputs_folder, "models"
)

os.makedirs(graphs_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)
os.makedirs(models_folder, exist_ok=True)

# ------------------------------------------------------------
# 15. SAVE MISSING VALUE REPORT
# ------------------------------------------------------------

missing_report.to_csv(
    os.path.join(
        results_folder,
        "missing_value_report.csv"
    )
)

# ------------------------------------------------------------
# 16. STATISTICAL SUMMARY
# ------------------------------------------------------------

print("\n========== STATISTICAL SUMMARY ==========")

print(df.describe(include="all").T)

# ------------------------------------------------------------
# 17. FINAL DATASET INFORMATION
# ------------------------------------------------------------

print("\n========== CLEANED DATASET ==========")

print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

print("\nFinal Column Names:")

print(df.columns.tolist())

# ------------------------------------------------------------
# 18. SAVE CLEANED DATASET
# ------------------------------------------------------------

cleaned_file = os.path.join(
    project_folder,
    "Student_Performance_Prediction_Cleaned.csv"
)

df.to_csv(
    cleaned_file,
    index=False
)

# ------------------------------------------------------------
# 19. COMPLETION MESSAGE
# ------------------------------------------------------------

print("\n==============================================")
print("DATA CLEANING COMPLETED SUCCESSFULLY!")
print("==============================================")

print("\nCleaned Dataset Saved At:")

print(cleaned_file)

print("\nMissing Value Report Saved In:")

print(results_folder)

print("\nOriginal Dataset Has NOT Been Modified.")
