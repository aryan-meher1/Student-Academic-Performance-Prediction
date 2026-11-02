
# ============================================================
# STUDENT ACADEMIC PERFORMANCE PREDICTION
# FINAL OUTPUT VERIFICATION
# ============================================================

import os
import pandas as pd

BASE_DIR = r"C:\Users\aryan\Downloads\student performance analysis"

RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
GRAPHS_DIR = os.path.join(BASE_DIR, "outputs", "graphs")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")

print("\n==========================================")
print("       PROJECT OUTPUT VERIFICATION")
print("==========================================")

# ------------------------------------------------------------
# 1. CHECK CLASSIFICATION RESULTS
# ------------------------------------------------------------

print("\n1. CLASSIFICATION RESULTS")

classification_file = os.path.join(
    RESULTS_DIR,
    "classification_model_comparison.csv"
)

if os.path.exists(classification_file):

    classification = pd.read_csv(classification_file)

    print("Status: FOUND")
    print("Models evaluated:", len(classification))
    print(classification.to_string(index=False))

    if len(classification) == 4:
        print("PASS: Exactly 4 classification models found.")
    else:
        print("WARNING: Expected 4 classification models.")

else:
    print("MISSING: Classification comparison CSV")

# ------------------------------------------------------------
# 2. CHECK REGRESSION RESULTS
# ------------------------------------------------------------

print("\n2. REGRESSION RESULTS")

regression_file = os.path.join(
    RESULTS_DIR,
    "regression_model_comparison.csv"
)

if os.path.exists(regression_file):

    regression = pd.read_csv(regression_file)

    print("Status: FOUND")
    print(regression.to_string(index=False))

    if (
        len(regression) == 1
        and regression.iloc[0]["Model"] == "Multiple Linear Regression"
    ):
        print("PASS: Only Multiple Linear Regression found.")
    else:
        print("WARNING: Check regression results.")

else:
    print("MISSING: Regression comparison CSV")

# ------------------------------------------------------------
# 3. CHECK CONFUSION MATRICES
# ------------------------------------------------------------

print("\n3. CLASSIFICATION GRAPHS")

confusion_graphs = [
    "classification_confusion_knn.png",
    "classification_confusion_naive_bayes.png",
    "classification_confusion_decision_tree.png",
    "classification_confusion_svm.png"
]

for graph in confusion_graphs:

    path = os.path.join(GRAPHS_DIR, graph)

    if os.path.exists(path):
        print("FOUND:", graph)
    else:
        print("MISSING:", graph)

# ------------------------------------------------------------
# 4. CHECK REGRESSION GRAPH
# ------------------------------------------------------------

print("\n4. REGRESSION GRAPH")

regression_graph = os.path.join(
    GRAPHS_DIR,
    "multiple_linear_regression.png"
)

if os.path.exists(regression_graph):
    print("FOUND: multiple_linear_regression.png")
else:
    print("MISSING: multiple_linear_regression.png")

# ------------------------------------------------------------
# 5. CHECK REGRESSION PREDICTIONS
# ------------------------------------------------------------

print("\n5. REGRESSION PREDICTIONS")

prediction_file = os.path.join(
    RESULTS_DIR,
    "regression_predictions.csv"
)

if os.path.exists(prediction_file):

    predictions = pd.read_csv(prediction_file)

    print("FOUND: regression_predictions.csv")
    print("Prediction records:", len(predictions))

else:
    print("MISSING: regression_predictions.csv")

# ------------------------------------------------------------
# 6. CHECK SAVED TRAINED MODELS
# ------------------------------------------------------------

print("\n6. TRAINED MODEL FILES")

if os.path.exists(MODELS_DIR):

    model_files = [
        file for file in os.listdir(MODELS_DIR)
        if file.endswith(".pkl")
    ]

    if model_files:
        for file in model_files:
            print("FOUND:", file)
    else:
        print("No saved model files found.")

else:
    print("Models directory does not exist.")

# ------------------------------------------------------------
# 7. CHECK UNWANTED CLUSTERING OUTPUTS
# ------------------------------------------------------------

print("\n7. OLD CLUSTERING OUTPUTS")

clustering_files = [
    "clustering_silhouette_scores.csv",
    "kmeans_cluster_assignments.csv",
    "hierarchical_cluster_sample.csv",
    "kmeans_cluster_profile.csv"
]

for file in clustering_files:

    path = os.path.join(RESULTS_DIR, file)

    if os.path.exists(path):
        print("OLD OUTPUT FOUND:", file)
    else:
        print("Not present:", file)

# ------------------------------------------------------------
# 8. FINAL MESSAGE
# ------------------------------------------------------------

print("\n==========================================")
print("       VERIFICATION FINISHED")
print("==========================================")

print("No files have been deleted or modified.")
print("Review any MISSING or OLD OUTPUT FOUND messages.")
