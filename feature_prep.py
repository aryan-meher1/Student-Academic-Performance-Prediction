
# ============================================================
# STUDENT ACADEMIC PERFORMANCE PREDICTION
# FILE 1: FEATURE PREPARATION
# ============================================================

import os
import pandas as pd

# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_FOLDER = r"C:\Users\aryan\Downloads\student performance analysis"

DATA_PATH = os.path.join(
    PROJECT_FOLDER,
    "Student_Performance_Prediction_Cleaned.csv"
)

OUTPUT_DIR = os.path.join(PROJECT_FOLDER, "outputs")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------
# 2. LOAD CLEANED DATASET
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\n======================================")
print("FEATURE PREPARATION STARTED")
print("======================================")

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# ------------------------------------------------------------
# 3. SELECT FEATURES FOR CLASSIFICATION
# Target: pass_fail
# ------------------------------------------------------------

classification_features = [
    "course_name",
    "gender",
    "region",
    "previous_education",
    "socioeconomic_group",
    "age_group",
    "previous_attempts",
    "registered_credits",
    "disability_status",
    "registration_days",
    "previous_marks",
    "active_study_days"
]

classification_target = "pass_fail"


# ------------------------------------------------------------
# 4. SELECT FEATURES FOR REGRESSION
# Target: final_assessment_marks
# ------------------------------------------------------------

regression_features = [
    "course_name",
    "gender",
    "region",
    "previous_education",
    "socioeconomic_group",
    "age_group",
    "previous_attempts",
    "registered_credits",
    "disability_status",
    "registration_days",
    "previous_marks",
    "active_study_days"
]

regression_target = "final_assessment_marks"


# ------------------------------------------------------------
# 5. CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = list(set(
    classification_features
    + regression_features
    + [classification_target, regression_target]
))

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing columns in dataset: " + str(missing_columns)
    )

print("\nAll required columns are available!")


# ------------------------------------------------------------
# 6. REMOVE EMPTY COURSE SESSION COLUMN
# ------------------------------------------------------------

if "course_session" in df.columns:
    if df["course_session"].isna().all():
        df.drop(columns=["course_session"], inplace=True)
        print("Empty course_session column removed.")


# ------------------------------------------------------------
# 7. PREPARE CLASSIFICATION DATA
# ------------------------------------------------------------

classification_data = df[
    classification_features + [classification_target]
].copy()

classification_data.dropna(
    subset=[classification_target],
    inplace=True
)

classification_data[classification_target] = (
    classification_data[classification_target]
    .astype(str)
    .str.strip()
    .str.upper()
)

classification_data = classification_data[
    classification_data[classification_target].isin(["PASS", "FAIL"])
]

print("\n========== CLASSIFICATION DATA ==========")

print("Target variable:", classification_target)
print("Number of input features:", len(classification_features))
print("Total records:", len(classification_data))

print("\nTarget distribution:")
print(classification_data[classification_target].value_counts())


# ------------------------------------------------------------
# 8. PREPARE REGRESSION DATA
# ------------------------------------------------------------

regression_data = df[
    regression_features + [regression_target]
].copy()

regression_data[regression_target] = pd.to_numeric(
    regression_data[regression_target],
    errors="coerce"
)

regression_data.dropna(
    subset=[regression_target],
    inplace=True
)

print("\n========== REGRESSION DATA ==========")

print("Target variable:", regression_target)
print("Number of input features:", len(regression_features))
print("Total records:", len(regression_data))

print("\nRegression target statistics:")
print(regression_data[regression_target].describe())


# ------------------------------------------------------------
# 9. SAVE PREPARED DATASETS
# ------------------------------------------------------------

classification_path = os.path.join(
    RESULTS_DIR,
    "classification_dataset.csv"
)

regression_path = os.path.join(
    RESULTS_DIR,
    "regression_dataset.csv"
)

classification_data.to_csv(
    classification_path,
    index=False
)

regression_data.to_csv(
    regression_path,
    index=False
)


# ------------------------------------------------------------
# 10. SAVE SELECTED FEATURE INFORMATION
# ------------------------------------------------------------

feature_info = pd.DataFrame({
    "Classification_Features": pd.Series(classification_features),
    "Regression_Features": pd.Series(regression_features)
})

feature_info.to_csv(
    os.path.join(RESULTS_DIR, "selected_features.csv"),
    index=False
)


# ------------------------------------------------------------
# 11. DISPLAY FINAL INFORMATION
# ------------------------------------------------------------

print("\n========== SAVED FILES ==========")

print("Classification dataset:")
print(classification_path)

print("\nRegression dataset:")
print(regression_path)

print("\nFeature information:")
print(os.path.join(RESULTS_DIR, "selected_features.csv"))

print("\n======================================")
print("FEATURE PREPARATION COMPLETED!")
print("======================================")
