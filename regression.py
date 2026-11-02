
# ============================================================
# STUDENT ACADEMIC PERFORMANCE PREDICTION
# FILE: REGRESSION.PY
# MODEL: MULTIPLE LINEAR REGRESSION ONLY
# ============================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. DEFINE PROJECT PATHS
# ============================================================

BASE_DIR = r"C:\Users\aryan\Downloads\student performance analysis"

DATA_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "results",
    "regression_dataset.csv"
)

RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
GRAPHS_DIR = os.path.join(BASE_DIR, "outputs", "graphs")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD PREPARED REGRESSION DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\n======================================")
print("MULTIPLE LINEAR REGRESSION STARTED")
print("======================================")

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 3. SELECT FEATURES AND TARGET
# ============================================================

target = "final_assessment_marks"

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

data = df[features + [target]].copy()

data[target] = pd.to_numeric(
    data[target],
    errors="coerce"
)

data.dropna(subset=[target], inplace=True)

X = data[features]
y = data[target]

print("\n========== FEATURE INFORMATION ==========")

print("Number of input features:", len(features))
print("Target variable:", target)
print("Available records:", len(data))


# ============================================================
# 4. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# ============================================================
# 5. PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ],
    sparse_threshold=0
)


# ============================================================
# 6. SPLIT DATA INTO TRAINING AND TESTING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n========== DATA SPLITTING ==========")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 7. CREATE MULTIPLE LINEAR REGRESSION MODEL
# ============================================================

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

print("\n========== MODEL TRAINING ==========")

model.fit(X_train, y_train)

print("Multiple Linear Regression trained successfully!")


# ============================================================
# 8. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 9. EVALUATE MODEL PERFORMANCE
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

print(f"Variance Explained: {r2 * 100:.2f}%")


# ============================================================
# 10. SAVE EVALUATION RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": ["Multiple Linear Regression"],
    "MAE": [mae],
    "MSE": [mse],
    "RMSE": [rmse],
    "R2_Score": [r2]
})

results_path = os.path.join(
    RESULTS_DIR,
    "regression_model_comparison.csv"
)

results.to_csv(
    results_path,
    index=False
)


# ============================================================
# 11. SAVE ACTUAL VS PREDICTED VALUES
# ============================================================

predictions = pd.DataFrame({
    "Actual_Score": y_test.values,
    "Predicted_Score": y_pred
})

predictions_path = os.path.join(
    RESULTS_DIR,
    "regression_predictions.csv"
)

predictions.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# 12. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Assessment Marks")
plt.ylabel("Predicted Assessment Marks")

plt.title("Multiple Linear Regression: Actual vs Predicted")

plt.tight_layout()

graph_path = os.path.join(
    GRAPHS_DIR,
    "multiple_linear_regression.png"
)

plt.savefig(
    graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. RESIDUAL PLOT
# ============================================================

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.5
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Assessment Marks")
plt.ylabel("Residuals")

plt.title("Residual Plot - Multiple Linear Regression")

plt.tight_layout()

residual_graph_path = os.path.join(
    GRAPHS_DIR,
    "regression_residual_plot.png"
)

plt.savefig(
    residual_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. SAVE TRAINED MODEL
# ============================================================

model_path = os.path.join(
    MODELS_DIR,
    "multiple_linear_regression.pkl"
)

joblib.dump(model, model_path)


# ============================================================
# 15. COMPLETION MESSAGE
# ============================================================

print("\n======================================")
print("MULTIPLE LINEAR REGRESSION COMPLETED!")
print("======================================")

print("\nEvaluation results:", results_path)
print("Predictions:", predictions_path)
print("Actual vs predicted graph:", graph_path)
print("Residual graph:", residual_graph_path)
print("Trained model:", model_path)

print("\n======================================")
