# ==========================
# IMPORT LIBRARIES
# ==========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("First 5 Rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nInfo:")
print(df.info())

# ==========================
# DATA CLEANING
# ==========================

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop missing values
df = df.dropna()

print("\nAfter Cleaning Shape:", df.shape)

# ==========================
# DATA VISUALIZATION
# ==========================

plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Contract vs Churn")
plt.xticks(rotation=20)
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(x="InternetService", hue="Churn", data=df)
plt.title("Internet Service vs Churn")
plt.show()

# ==========================
# ENCODING
# ==========================

binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]

for col in binary_cols:
    df[col] = df[col].map({"Yes":1, "No":0})

# Remove customerID
df = df.drop("customerID", axis=1)

# One hot encoding
df = pd.get_dummies(df, drop_first=True)

print("\nShape after encoding:", df.shape)

# ==========================
# SPLIT FEATURES AND TARGET
# ==========================

X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nX shape:", X.shape)
print("y shape:", y.shape)

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ==========================
# FEATURE SCALING (IMPORTANT FIX)
# ==========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# ==========================
# LOGISTIC REGRESSION MODEL
# ==========================

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

logistic_accuracy = accuracy_score(y_test, y_pred)

print("\nLOGISTIC REGRESSION RESULTS")
print(f"Accuracy: {logistic_accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================
# DECISION TREE MODEL
# ==========================

tree = DecisionTreeClassifier(random_state=42)

tree.fit(X_train, y_train)

tree_pred = tree.predict(X_test)

tree_accuracy = accuracy_score(y_test, tree_pred)

print("\nDECISION TREE RESULTS")
print(f"Accuracy: {tree_accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, tree_pred))

print("\nClassification Report:")
print(classification_report(y_test, tree_pred))
# ==========================
# RANDOM FOREST (TUNED)
# ==========================

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)

rf_model = grid.best_estimator_

rf_pred = rf_model.predict(X_test_scaled)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\n============================")
print("RANDOM FOREST (TUNED)")
print("============================")

print("Best Parameters:")
print(grid.best_params_)

print(f"\nAccuracy: {rf_accuracy*100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))
# ==========================
# CROSS VALIDATION
# ==========================

cv_scores = cross_val_score(
    rf_model,
    X_train_scaled,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\n============================")
print("CROSS VALIDATION RESULTS")
print("============================")

print("Cross Validation Scores:")
print(cv_scores)

print(f"\nAverage CV Accuracy: {cv_scores.mean() * 100:.2f}%")
print(f"Standard Deviation: {cv_scores.std() * 100:.2f}%")

# ==========================
# FEATURE IMPORTANCE
# ==========================

importances = rf_model.feature_importances_

feat_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
})

feat_df = feat_df.sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(feat_df.head(10))

plt.figure(figsize=(10,6))
sns.barplot(
    data=feat_df.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()


# ==========================
# SAVE MODELS
# ==========================

joblib.dump(model, "logistic_model.pkl")
joblib.dump(rf_model, "random_forest.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")
print("\nModels saved successfully!")

loaded_model = joblib.load("random_forest.pkl")
print("Model loaded successfully!")


# ==========================
# MODEL COMPARISON
# ==========================

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, tree_pred),
        accuracy_score(y_test, rf_pred)
    ],
    "Precision": [
        classification_report(y_test, y_pred, output_dict=True)["1"]["precision"],
        classification_report(y_test, tree_pred, output_dict=True)["1"]["precision"],
        classification_report(y_test, rf_pred, output_dict=True)["1"]["precision"]
    ],
    "Recall": [
        classification_report(y_test, y_pred, output_dict=True)["1"]["recall"],
        classification_report(y_test, tree_pred, output_dict=True)["1"]["recall"],
        classification_report(y_test, rf_pred, output_dict=True)["1"]["recall"]
    ],
    "F1 Score": [
        classification_report(y_test, y_pred, output_dict=True)["1"]["f1-score"],
        classification_report(y_test, tree_pred, output_dict=True)["1"]["f1-score"],
        classification_report(y_test, rf_pred, output_dict=True)["1"]["f1-score"]
    ]
})

comparison["Accuracy"] *= 100
comparison["Precision"] *= 100
comparison["Recall"] *= 100
comparison["F1 Score"] *= 100

comparison = comparison.round(2)

print("\n============================")
print("MODEL COMPARISON")
print("============================")
print(comparison.to_string(index=False))

plt.figure(figsize=(8,5))

sns.barplot(
    data=comparison,
    x="Model",
    y="Accuracy"
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")

plt.show()
