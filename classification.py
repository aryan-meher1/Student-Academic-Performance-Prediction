
# ============================================================
# STUDENT ACADEMIC PERFORMANCE PREDICTION
# FILE 2: CLASSIFICATION MODELS
# MODELS: KNN, NAIVE BAYES, DECISION TREE, SVM
# ============================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = r"C:\Users\aryan\Downloads\student performance analysis"

DATA_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "results",
    "classification_dataset.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
GRAPHS_DIR = os.path.join(OUTPUT_DIR, "graphs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD PREPARED CLASSIFICATION DATASET
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\n======================================")
print("CLASSIFICATION MODEL TRAINING STARTED")
print("======================================")

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# ------------------------------------------------------------
# 3. SELECT INPUT FEATURES AND TARGET
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

target = "pass_fail"

X = df[features].copy()
y = df[target].copy()

print("\nInput features:", len(features))
print("Target variable:", target)

print("\nTarget distribution:")
print(y.value_counts())


# ------------------------------------------------------------
# 4. IDENTIFY CATEGORICAL AND NUMERICAL FEATURES
# ------------------------------------------------------------

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numeric_features)


# ------------------------------------------------------------
# 5. CREATE PREPROCESSING PIPELINES
# ------------------------------------------------------------

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ],
    sparse_threshold=0
)


# ------------------------------------------------------------
# 6. SPLIT DATA INTO TRAINING AND TESTING SETS
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN-TEST SPLIT ==========")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ------------------------------------------------------------
# 7. DEFINE FOUR CLASSIFICATION MODELS
# ------------------------------------------------------------

models = {

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Naive Bayes": GaussianNB(),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=12,
        class_weight="balanced",
        random_state=42
    ),

    "SVM": LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=5000
    )

}


# ------------------------------------------------------------
# 8. TRAIN AND EVALUATE MODELS
# ------------------------------------------------------------

results = []

for model_name, model in models.items():

    print("\n======================================")
    print("TRAINING MODEL:", model_name)
    print("======================================")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # TRAIN MODEL
    pipeline.fit(X_train, y_train)

    # PREDICT TEST DATA
    predictions = pipeline.predict(X_test)

    # CALCULATE PERFORMANCE METRICS
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("\nAccuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    # --------------------------------------------------------
    # SAVE TRAINED MODEL
    # --------------------------------------------------------

    model_file = (
        model_name.lower().replace(" ", "_") + ".pkl"
    )

    model_path = os.path.join(
        MODELS_DIR,
        model_file
    )

    joblib.dump(pipeline, model_path)

    print("\nModel saved:", model_path)

    # --------------------------------------------------------
    # GENERATE CONFUSION MATRIX
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 6))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        ax=ax,
        xticks_rotation=45,
        cmap="Blues"
    )

    ax.set_title("Confusion Matrix - " + model_name)

    plt.tight_layout()

    graph_name = (
        "classification_confusion_"
        + model_name.lower().replace(" ", "_")
        + ".png"
    )

    plt.savefig(
        os.path.join(GRAPHS_DIR, graph_name),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ------------------------------------------------------------
# 9. COMPARE ALL CLASSIFICATION MODELS
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

comparison_path = os.path.join(
    RESULTS_DIR,
    "classification_model_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)

print("\n======================================")
print("CLASSIFICATION MODEL COMPARISON")
print("======================================")

print(results_df.to_string(index=False))


# ------------------------------------------------------------
# 10. GENERATE MODEL COMPARISON GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

plt.title("Classification Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPHS_DIR,
        "classification_model_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 11. COMPLETION MESSAGE
# ------------------------------------------------------------

print("\n======================================")
print("CLASSIFICATION COMPLETED SUCCESSFULLY!")
print("======================================")

print("\nTotal classification models trained:", len(models))

print("Trained models saved in:", MODELS_DIR)

print("Comparison saved in:", comparison_path)

print("Confusion matrices saved in:", GRAPHS_DIR)

print("\n======================================")
