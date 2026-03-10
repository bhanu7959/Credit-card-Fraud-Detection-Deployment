
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

MODEL_PATH = "best_fraud_model.pkl"
FEATURES_PATH = "feature_names.json"

# -----------------------------
# Load model and features
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Please keep best_fraud_model.pkl in the same folder as app.py"
        )
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_feature_names():
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "r") as f:
            return json.load(f)
    return [
        "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
        "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
        "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27",
        "V28", "Amount"
    ]

try:
    model = load_model()
    expected_features = load_feature_names()
except Exception as e:
    st.error(f"Startup error: {e}")
    st.stop()

# -----------------------------
# Sidebar model info
# -----------------------------
st.sidebar.title("📌 Model Info")
st.sidebar.write("**Model file:** best_fraud_model.pkl")
st.sidebar.write("**Prediction target:** Fraud / Non-Fraud")
st.sidebar.write("**Expected feature count:**", len(expected_features))
st.sidebar.write("**Expected columns:**")
st.sidebar.caption(", ".join(expected_features))

threshold = st.sidebar.slider(
    "Prediction Threshold",
    min_value=0.00,
    max_value=1.00,
    value=0.50,
    step=0.01,
    help="Transactions with fraud probability >= threshold will be flagged as fraud."
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Higher threshold = fewer fraud alerts.\n\n"
    "Lower threshold = more sensitive fraud detection."
)

# -----------------------------
# Main title
# -----------------------------
st.title("💳 Credit Card Fraud Detection Dashboard")
st.write("Upload a CSV file to predict fraudulent transactions and analyze suspicious activity.")

# -----------------------------
# Safer column validation
# -----------------------------
def validate_and_prepare_dataframe(df, expected_cols):
    uploaded_cols = list(df.columns)

    missing_cols = [col for col in expected_cols if col not in uploaded_cols]
    extra_cols = [col for col in uploaded_cols if col not in expected_cols]

    if missing_cols:
        return None, False, missing_cols, extra_cols, "missing"

    # Reorder only expected columns, safely ignore extras
    prepared_df = df.copy()[expected_cols]

    # Convert all columns to numeric safely
    for col in expected_cols:
        prepared_df[col] = pd.to_numeric(prepared_df[col], errors="coerce")

    null_counts = prepared_df.isnull().sum()
    bad_cols = null_counts[null_counts > 0]

    if len(bad_cols) > 0:
        return prepared_df, False, bad_cols.to_dict(), extra_cols, "invalid_values"

    return prepared_df, True, [], extra_cols, "ok"

# -----------------------------
# Risk label helper
# -----------------------------
def risk_label(prob):
    if prob >= 0.90:
        return "Very High"
    elif prob >= 0.70:
        return "High"
    elif prob >= 0.40:
        return "Medium"
    else:
        return "Low"

# -----------------------------
# Row highlighting
# -----------------------------
def highlight_fraud_rows(row):
    if row["Prediction"] == 1:
        return ["background-color: #ffe6e6; font-weight: bold; color: #900;"] * len(row)
    return [""] * len(row)

# -----------------------------
# File upload
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(raw_df.head(), use_container_width=True)

    prepared_df, is_valid, details, extra_cols, status = validate_and_prepare_dataframe(raw_df, expected_features)

    if extra_cols:
        st.warning(f"Extra columns found and ignored: {extra_cols}")

    if not is_valid:
        if status == "missing":
            st.error("Missing required columns.")
            st.write(details)
        elif status == "invalid_values":
            st.error("Some required columns contain missing or non-numeric values after conversion.")
            st.write(details)
        st.stop()

    if st.button("🚀 Run Prediction", use_container_width=True):
        with st.spinner("Running fraud detection model and generating dashboard..."):
            probabilities = model.predict_proba(prepared_df)[:, 1]
            predictions = (probabilities >= threshold).astype(int)

            result_df = raw_df.copy()
            result_df["Fraud_Probability"] = probabilities
            result_df["Prediction"] = predictions
            result_df["Risk_Level"] = [risk_label(p) for p in probabilities]

        # -----------------------------
        # Fraud summary metrics
        # -----------------------------
        total_transactions = len(result_df)
        fraud_transactions = int(result_df["Prediction"].sum())
        non_fraud_transactions = total_transactions - fraud_transactions
        avg_fraud_prob = float(result_df["Fraud_Probability"].mean())
        max_fraud_prob = float(result_df["Fraud_Probability"].max())

        st.subheader("📊 Fraud Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", total_transactions)
        col2.metric("Fraud Detected", fraud_transactions)
        col3.metric("Non-Fraud", non_fraud_transactions)
        col4.metric("Average Fraud Probability", f"{avg_fraud_prob:.4f}")

        # -----------------------------
        # Fraud risk alert
        # -----------------------------
        if fraud_transactions == 0:
            st.success("✅ No transactions crossed the selected fraud threshold.")
        else:
            fraud_rate = fraud_transactions / total_transactions
            if fraud_rate >= 0.20 or max_fraud_prob >= 0.95:
                st.error(
                    f"🚨 High Fraud Risk Alert: {fraud_transactions} transactions flagged. "
                    f"Highest fraud probability = {max_fraud_prob:.4f}"
                )
            elif fraud_rate >= 0.05:
                st.warning(
                    f"⚠️ Moderate Fraud Risk Alert: {fraud_transactions} suspicious transactions detected."
                )
            else:
                st.info(
                    f"🔎 Low-to-Moderate Risk: {fraud_transactions} suspicious transactions detected."
                )

        # -----------------------------
        # Fraud distribution chart
        # -----------------------------
        st.subheader("📈 Fraud Distribution Chart")
        counts = result_df["Prediction"].value_counts().sort_index()
        nonfraud_count = int(counts.get(0, 0))
        fraud_count = int(counts.get(1, 0))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Non-Fraud", "Fraud"], [nonfraud_count, fraud_count])
        ax.set_ylabel("Number of Transactions")
        ax.set_title("Fraud vs Non-Fraud Predictions")
        st.pyplot(fig)

        # -----------------------------
        # Top suspicious transactions
        # -----------------------------
        st.subheader("🔝 Top Suspicious Transactions")
        top_suspicious = result_df.sort_values("Fraud_Probability", ascending=False).head(10)
        st.dataframe(top_suspicious, use_container_width=True)

        # -----------------------------
        # Highlighted fraud rows
        # -----------------------------
        st.subheader("🎯 Prediction Results (Fraud Rows Highlighted)")
        styled_df = result_df.style.apply(highlight_fraud_rows, axis=1)
        st.dataframe(styled_df, use_container_width=True)

        # -----------------------------
        # Separate suspicious-only section
        # -----------------------------
        fraud_only_df = result_df[result_df["Prediction"] == 1].sort_values(
            "Fraud_Probability", ascending=False
        )

        if not fraud_only_df.empty:
            st.subheader("🚨 Suspicious Transactions Only")
            st.dataframe(fraud_only_df, use_container_width=True)

        # -----------------------------
        # Download predictions button
        # -----------------------------
        csv_data = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Predictions",
            data=csv_data,
            file_name="fraud_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.info("Please upload a CSV file with the required transaction features to begin.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "Built with Streamlit for Credit Card Fraud Detection | "
    "Upload transaction data, adjust threshold, review fraud risk, and download predictions."
)
