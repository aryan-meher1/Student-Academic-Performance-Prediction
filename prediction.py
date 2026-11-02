
# ============================================================
# STUDENT ACADEMIC PERFORMANCE PREDICTION
# FILE: PREDICTION.PY
# MODEL: SUPPORT VECTOR MACHINE (SVM)
# ============================================================

import os
import joblib
import pandas as pd

# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = r"C:\Users\aryan\Downloads\student performance analysis"

DATA_PATH = os.path.join(
    BASE_DIR,
    "Student_Performance_Prediction_Cleaned.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "models",
    "svm.pkl"
)

# ------------------------------------------------------------
# 2. LOAD DATASET AND TRAINED MODEL
# ------------------------------------------------------------

data = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

print("\n======================================")
print("STUDENT ACADEMIC PERFORMANCE PREDICTION")
print("======================================")

# ------------------------------------------------------------
# 3. GET TYPICAL VALUES FOR OTHER FEATURES
# ------------------------------------------------------------

student = {}

categorical = [
    "course_name",
    "gender",
    "region",
    "previous_education",
    "socioeconomic_group",
    "age_group",
    "disability_status"
]

numerical = [
    "previous_attempts",
    "registered_credits",
    "registration_days",
    "previous_marks",
    "active_study_days"
]

for col in categorical:
    student[col] = data[col].mode()[0]

for col in numerical:
    student[col] = data[col].median()

# ------------------------------------------------------------
# 4. ASK STUDENT DETAILS
# ------------------------------------------------------------

print("\nEnter Student Details:\n")

student["previous_attempts"] = int(
    input("Previous attempts: ")
)

student["registered_credits"] = int(
    input("Registered credits: ")
)

student["previous_marks"] = float(
    input("Previous marks: ")
)

student["active_study_days"] = int(
    input("Active study days: ")
)

student["registration_days"] = float(
    input("Registration days: ")
)

# ------------------------------------------------------------
# 5. ARRANGE INPUT FEATURES
# ------------------------------------------------------------

features = [
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

input_data = pd.DataFrame(
    [student],
    columns=features
)

# ------------------------------------------------------------
# 6. PREDICT ACADEMIC RESULT
# ------------------------------------------------------------

prediction = model.predict(input_data)[0]

# ------------------------------------------------------------
# 7. DISPLAY PREDICTION
# ------------------------------------------------------------

print("\n======================================")
print("          PREDICTION RESULT")
print("======================================")

print("Predicted Academic Result:", prediction)

if prediction == "PASS":
    print("\nThe model predicts that the student will PASS.")
else:
    print("\nThe model predicts that the student may FAIL.")

print("\n======================================")
