# Deploy Your Model as a Live Web App 🩺

A simple, shareable **Streamlit web app** that predicts whether a breast
tumor is benign or malignant, powered by a Logistic Regression model
trained earlier on the Breast Cancer Wisconsin dataset.

## 🔗 Live App

**[Click here to try the live app](https://model-deployment-webapp-7n2rmuvn4fe2cg5lym2hco.streamlit.app/)**

*(Replace the link above with your actual Streamlit Community Cloud URL after deploying — see steps below.)*

## What this app does

- Takes 10 key tumor measurements as input (mean radius, mean texture, mean
  perimeter, mean area, mean smoothness, mean compactness, mean concavity,
  mean concave points, mean symmetry, mean fractal dimension)
- Loads a pre-trained Logistic Regression model (saved with `joblib`)
- Predicts **Benign** or **Malignant** with a confidence score when you
  click **Predict**
- Includes a "Load Example Values" button so anyone can try it instantly
  without needing real medical data

## Model

Trained on the built-in `sklearn` Breast Cancer Wisconsin dataset, using
only the 10 "mean" features (not all 30) to keep the web form simple and
user-friendly. Test accuracy: **~89%**.

The model was trained in `train_and_save_model.py` and saved as
`model.joblib` (plus `feature_names.joblib` for the exact feature order).

## Run locally

```bash
pip install -r requirements.txt
python train_and_save_model.py   # trains + saves model.joblib (already included, optional to re-run)
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## How this was deployed (Streamlit Community Cloud)

1. Pushed this repo to GitHub (public repo, includes `app.py`,
   `model.joblib`, `feature_names.joblib`, `requirements.txt`).
2. Went to [share.streamlit.io](https://share.streamlit.io), signed in with
   GitHub.
3. Clicked **"New app"**, selected this repository, branch `main`, and set
   the main file path to `app.py`.
4. Clicked **Deploy** — Streamlit Cloud installed everything from
   `requirements.txt` and built the live app automatically.
5. Copied the live app URL and added it to the top of this README.

## Files

```
model-deployment-webapp/
├── app.py                    # Streamlit web app
├── train_and_save_model.py   # Trains and saves the model
├── model.joblib               # Saved trained model
├── feature_names.joblib       # Saved feature name order
├── sample_inputs.csv          # A few example rows to test the app with
├── requirements.txt
└── README.md
```

> ⚠️ **Disclaimer:** This is a machine learning demo project, not a real
> medical diagnostic tool. Predictions should never be used for actual
> medical decisions.
