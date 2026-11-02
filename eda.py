# ==========================================================
# STUDENT ACADEMIC PERFORMANCE PREDICTION
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("\n======================================")
print("EXPLORATORY DATA ANALYSIS STARTED")
print("======================================")

# ----------------------------------------------------------
# 1. LOAD CLEANED DATASET
# ----------------------------------------------------------

df = pd.read_csv("Student_Performance_Prediction_Cleaned.csv")

print("\nDataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# Create output folders
os.makedirs("outputs/graphs", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

print("\nMissing Values:")
print(df.isnull().sum())

# Remove completely empty course_session column
if "course_session" in df.columns and df["course_session"].isnull().all():
    df = df.drop(columns=["course_session"])
    print("\nEmpty course_session column removed.")

# ==========================================================
# GRAPH 1: PASS/FAIL DISTRIBUTION
# ==========================================================

plt.figure(figsize=(9, 6))

sns.countplot(
    data=df,
    x="pass_fail",
    order=["PASS", "FAIL"]
)

plt.title("Distribution of Student Pass and Fail Results", fontsize=15)
plt.xlabel("Academic Result")
plt.ylabel("Number of Students")

plt.tight_layout()
plt.savefig("outputs/graphs/graph1_pass_fail_distribution.png", dpi=300)
plt.show()

print("\n========== PASS/FAIL DISTRIBUTION ==========")

result_summary = pd.DataFrame({
    "Student_Count": df["pass_fail"].value_counts(),
    "Percentage": (
        df["pass_fail"].value_counts(normalize=True) * 100
    ).round(2)
})

print(result_summary)

# ==========================================================
# GRAPH 2: ACADEMIC PERFORMANCE DISTRIBUTION
# ==========================================================

performance_order = ["Excellent", "Good", "Average", "Poor"]

plt.figure(figsize=(9, 6))

sns.countplot(
    data=df,
    x="academic_performance",
    order=performance_order
)

plt.title("Distribution of Academic Performance", fontsize=15)
plt.xlabel("Academic Performance")
plt.ylabel("Number of Students")

plt.tight_layout()
plt.savefig("outputs/graphs/graph2_academic_performance_distribution.png", dpi=300)
plt.show()

print("\n========== ACADEMIC PERFORMANCE SUMMARY ==========")

performance_summary = pd.DataFrame({
    "Student_Count": df["academic_performance"].value_counts(),
    "Percentage": (
        df["academic_performance"].value_counts(normalize=True) * 100
    ).round(2)
})

print(performance_summary)

# ==========================================================
# GRAPH 3: ASSESSMENT MARKS DISTRIBUTION
# ==========================================================

plt.figure(figsize=(9, 6))

sns.histplot(
    data=df,
    x="assessment_marks",
    bins=30,
    kde=True
)

plt.title("Distribution of Student Assessment Marks", fontsize=15)
plt.xlabel("Assessment Marks")
plt.ylabel("Number of Students")

plt.tight_layout()
plt.savefig("outputs/graphs/graph3_assessment_marks_distribution.png", dpi=300)
plt.show()

print("\n========== ASSESSMENT MARK STATISTICS ==========")

print(df["assessment_marks"].describe())

# ==========================================================
# GRAPH 4: ASSESSMENT MARKS BY PASS/FAIL RESULT
# ==========================================================

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="pass_fail",
    y="assessment_marks",
    order=["PASS", "FAIL"]
)

plt.title("Assessment Marks by Pass/Fail Result", fontsize=15)
plt.xlabel("Academic Result")
plt.ylabel("Assessment Marks")

plt.tight_layout()
plt.savefig("outputs/graphs/graph4_marks_by_pass_fail.png", dpi=300)
plt.show()

print("\n========== MARKS BY PASS/FAIL RESULT ==========")

marks_comparison = df.groupby(
    "pass_fail"
)["assessment_marks"].agg(
    ["count", "mean", "median", "min", "max"]
)

print(marks_comparison.round(2))

# ==========================================================
# GRAPH 5: ACTIVE STUDY DAYS VS ASSESSMENT MARKS
# ==========================================================

activity_data = df[
    ["active_study_days", "assessment_marks"]
].dropna()

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=activity_data,
    x="active_study_days",
    y="assessment_marks",
    alpha=0.25,
    s=18
)

plt.title("Active Study Days vs Assessment Marks", fontsize=15)
plt.xlabel("Active Study Days")
plt.ylabel("Assessment Marks")

plt.tight_layout()
plt.savefig("outputs/graphs/graph5_study_days_vs_marks.png", dpi=300)
plt.show()

activity_correlation = activity_data[
    "active_study_days"
].corr(activity_data["assessment_marks"])

print("\n========== STUDY ACTIVITY ANALYSIS ==========")

print("Students Included:", len(activity_data))
print("Correlation:", round(activity_correlation, 3))

# ==========================================================
# GRAPH 6: PREVIOUS ATTEMPTS VS PASS/FAIL RESULT
# ==========================================================

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="previous_attempts",
    hue="pass_fail",
    hue_order=["PASS", "FAIL"]
)

plt.title("Previous Attempts vs Pass/Fail Results", fontsize=15)
plt.xlabel("Number of Previous Attempts")
plt.ylabel("Number of Students")

plt.legend(title="Academic Result")

plt.tight_layout()
plt.savefig("outputs/graphs/graph6_previous_attempts_vs_result.png", dpi=300)
plt.show()

print("\n========== PREVIOUS ATTEMPTS ANALYSIS ==========")

attempt_summary = pd.crosstab(
    df["previous_attempts"],
    df["pass_fail"]
)

print(attempt_summary)

# ==========================================================
# GRAPH 7: CORRELATION HEATMAP
# ==========================================================

correlation_columns = [
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

correlation_data = df[correlation_columns].corr()

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_data,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Between Numerical Student Features", fontsize=15)

plt.tight_layout()
plt.savefig("outputs/graphs/graph7_correlation_heatmap.png", dpi=300)
plt.show()

print("\n========== CORRELATION MATRIX ==========")

print(correlation_data.round(3))

# ==========================================================
# SAVE EDA RESULTS
# ==========================================================

result_summary.to_csv("outputs/results/pass_fail_summary.csv")

performance_summary.to_csv(
    "outputs/results/academic_performance_summary.csv"
)

marks_comparison.to_csv(
    "outputs/results/assessment_marks_summary.csv"
)

attempt_summary.to_csv(
    "outputs/results/previous_attempts_summary.csv"
)

correlation_data.to_csv(
    "outputs/results/correlation_matrix.csv"
)

print("\n======================================")
print("EDA COMPLETED SUCCESSFULLY!")
print("======================================")

print("\nTotal Graphs Generated: 7")
print("Graphs Saved In: outputs/graphs/")
print("Analysis Saved In: outputs/results/")