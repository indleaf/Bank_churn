import streamlit as st
import pandas as pd
import pickle
from pre_pro_v2 import build_feature_matrix_simple  # simple scaling + OHE

st.set_page_config(page_title="Customer Input Form", layout="centered")
st.title("📋 Customer Profile Input")

# --- Load model (supports plain .pkl OR bundle {"model","feature_names"}) ---
@st.cache_resource
def load_model(path: str = "xgb_model.pkl"):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj.get("feature_names")
    return obj, None

# Change the filename here if you saved as xgb_bundle.pkl
model, feature_names = load_model("xgb_model.pkl")

# --- Inputs ---
geography = st.selectbox("🌍 Select Geography", ["France", "Germany", "Spain"])
surname = st.text_input("🧑 Surname", placeholder="Enter surname")

credit_score = st.number_input("💳 Credit Score", min_value=0, max_value=850, value=600)
gender = st.selectbox("⚧️ Gender", ["Male", "Female"])
age = st.number_input("🎂 Age", min_value=18.0, max_value=100.0, value=35.0)
tenure = st.number_input("📅 Tenure (Years)", min_value=0, max_value=10, value=3)

balance = st.number_input("💰 Balance", min_value=0.0, value=260000.0)
num_products = st.number_input("📦 Number of Products", min_value=1, max_value=4, value=2, step=1)
has_cr_card = st.selectbox("💳 Has Credit Card?", [1, 0])   # ints to avoid float->int surprises
is_active = st.selectbox("🔄 Is Active Member?", [1, 0])
estimated_salary = st.number_input("💼 Estimated Salary", min_value=0.0, value=200000.0)

row = {
    # Only these are used by build_feature_matrix_simple:
    "CreditScore": int(credit_score),
    "Age": float(age),
    "Balance": float(balance),
    "EstimatedSalary": float(estimated_salary),
    "Geography": geography,
    "Gender": gender,
    "NumOfProducts": int(num_products),

    # Extras (ignored by simple FE but kept for completeness/expansion)
    "Surname": surname,
    "Tenure": int(tenure),
    "HasCrCard": int(has_cr_card),
    "IsActiveMember": int(is_active),
}

# Keep latest input
if "latest_df" not in st.session_state:
    st.session_state.latest_df = pd.DataFrame(columns=row.keys())

if st.button("Submit"):
    df = pd.DataFrame([row])
    st.session_state.latest_df = df
    st.success("✅ Input submitted!")
    st.dataframe(df, use_container_width=True)

df_for_ml = st.session_state.latest_df

# --- Prediction ---
if df_for_ml.empty:
    st.info("Enter values and click Submit to run a prediction.")
else:
    # 1) Build features with the simple FE function
    X = build_feature_matrix_simple(df_for_ml)

    # 2) Determine expected columns and align (fixes OHE-mismatch errors)
    expected_cols = feature_names
    if expected_cols is None:
        expected_cols = getattr(model, "feature_names_in_", None)
    if expected_cols is None and hasattr(model, "get_booster"):
        expected_cols = model.get_booster().feature_names

    if expected_cols is not None:
        X_in = X.reindex(columns=expected_cols, fill_value=0)
    else:
        X_in = X  # assume exact match (not recommended unless you’re sure)

# 3) Predict (Exit vs Stay)
st.subheader("🔮 Prediction")

LABELS = {0: "Stay", 1: "Exit"}  # assuming target Exited: 1=Exit, 0=Stay
THRESHOLD = 0.5

if hasattr(model, "predict_proba"):  # classifier path
    proba = model.predict_proba(X_in)[:, 1]          # P(Exit)
    y_hat = (proba >= THRESHOLD).astype(int)         # 1=Exit
    st.metric("Exit probability", f"{proba[0]:.2%}")
    st.write("Prediction:", f"**{LABELS[int(y_hat[0])]}**")
else:  # models without predict_proba (or regressors returning P(Exit))
    y_pred = model.predict(X_in)
    # If y_pred is probability-like, threshold it; if it's class, cast to int.
    try:
        p_exit = float(y_pred[0])
        y_hat = int(p_exit >= THRESHOLD)
        st.metric("Exit probability (model output)", f"{p_exit:.2%}")
        st.write("Prediction:", f"**{LABELS[y_hat]}**")
    except Exception:
        cls = int(y_pred[0])
        st.write("Prediction:", f"**{LABELS.get(cls, cls)}**")

with st.expander("Show feature vector used for prediction"):
    st.dataframe(X_in, use_container_width=True)

