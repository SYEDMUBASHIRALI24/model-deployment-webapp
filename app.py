"""
Streamlit web app: Breast Cancer Prediction
Loads the saved Logistic Regression model (model.joblib) and lets a user
enter the 10 key "mean" tumor measurements to get a prediction.

Run locally with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🩺", layout="centered")

# ---------------------------------------------------------
# Load model + feature names
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.joblib")
    feature_names = joblib.load("feature_names.joblib")
    return model, feature_names

model, feature_names = load_model()

# ---------------------------------------------------------
# Page header
# ---------------------------------------------------------
st.title("🩺 Breast Cancer Prediction App")
st.write(
    "This app uses a **Logistic Regression** model (trained on the "
    "Breast Cancer Wisconsin dataset) to predict whether a tumor is "
    "**benign** or **malignant**, based on 10 measured cell characteristics. "
    "Enter values below, or click *Load Example* to auto-fill sample data."
)

# Reasonable default/example values (roughly dataset averages)
DEFAULTS = {
    "mean radius": 14.1,
    "mean texture": 19.3,
    "mean perimeter": 92.0,
    "mean area": 655.0,
    "mean smoothness": 0.096,
    "mean compactness": 0.104,
    "mean concavity": 0.089,
    "mean concave points": 0.049,
    "mean symmetry": 0.181,
    "mean fractal dimension": 0.063,
}

if "values" not in st.session_state:
    st.session_state.values = DEFAULTS.copy()

if st.button("🔄 Load Example Values"):
    st.session_state.values = DEFAULTS.copy()

# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
st.subheader("Tumor Measurements")
col1, col2 = st.columns(2)

user_input = {}
for i, feature in enumerate(feature_names):
    col = col1 if i % 2 == 0 else col2
    user_input[feature] = col.number_input(
        feature.title(),
        min_value=0.0,
        value=float(st.session_state.values.get(feature, 0.0)),
        step=0.01,
        format="%.4f",
    )

# ---------------------------------------------------------
# Predict button
# ---------------------------------------------------------
if st.button("🔍 Predict", type="primary"):
    input_df = pd.DataFrame([user_input])[feature_names]
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.divider()
    if prediction == 1:
        st.success(f"### ✅ Prediction: Benign\nConfidence: {probability[1]*100:.1f}%")
    else:
        st.error(f"### ⚠️ Prediction: Malignant\nConfidence: {probability[0]*100:.1f}%")

    st.caption(
        "Note: This is a demo app for a machine learning project, "
        "not a medical diagnostic tool. Always consult a doctor for real diagnoses."
    )

st.divider()
st.caption("Model: Logistic Regression | Dataset: Breast Cancer Wisconsin (sklearn) | Built with Streamlit")
