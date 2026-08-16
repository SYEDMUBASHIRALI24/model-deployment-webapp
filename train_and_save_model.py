"""
Train and save the best-performing model (Logistic Regression, ~98% accuracy
on the earlier "Ensemble Learning" task) so it can be loaded inside a
Streamlit web app.

We use only the 10 "mean" features (out of the original 30) from the Breast
Cancer Wisconsin dataset, so the web app form has a manageable number of
input fields for a user to fill in.
"""

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# 1. Load data, keep only the 10 "mean" features
# ---------------------------------------------------------
data = load_breast_cancer()
X_full = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
]
X = X_full[FEATURES]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 2. Train model (Logistic Regression, scaled — best performer earlier)
# ---------------------------------------------------------
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000, random_state=42)
)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Test accuracy with 10 mean-features model: {acc:.4f}")

# ---------------------------------------------------------
# 3. Save model with joblib
# ---------------------------------------------------------
joblib.dump(model, "model.joblib")
joblib.dump(FEATURES, "feature_names.joblib")
print("Saved: model.joblib")
print("Saved: feature_names.joblib")

# Save a few sample rows (for reference / demo values in the app)
sample = X_test.copy()
sample["target"] = y_test.values
sample.head(5).to_csv("sample_inputs.csv", index=False)
print("Saved: sample_inputs.csv (example rows you can use to test the app)")
