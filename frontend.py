
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Page settings
st.set_page_config(
    page_title="FIN-SHIELD",
    page_icon="🛡️",
    layout="centered"
)

MODEL_REPO = "kamrandar1911/fin-shield-model"
MODEL_FILE = "fin_shield_creditcard_model.joblib"


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE
    )
    return joblib.load(model_path)


st.title("🛡️ FIN-SHIELD")
st.subheader("AI-Based Credit Card Fraud Detection")
st.write(
    "Enter transaction features to check the model's prediction."
)

with st.spinner("Loading fraud detection model..."):
    try:
        model = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

st.warning(
    "V1–V28 are anonymized PCA features from the original dataset. "
    "Use real, correctly transformed transaction data for meaningful predictions."
)

with st.form("transaction_form"):
    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0,
        step=10.0
    )

    transaction_time = st.number_input(
        "Transaction Time (seconds from dataset reference)",
        min_value=0.0,
        value=0.0,
        step=1.0
    )

    st.subheader("Anonymized Features (V1–V28)")
    features = {}

    cols = st.columns(2)
    for i in range(1, 29):
        with cols[(i - 1) % 2]:
            features[f"V{i}"] = st.number_input(
                f"V{i}",
                value=0.0,
                step=0.1,
                format="%.4f",
                key=f"v{i}"
            )

    submitted = st.form_submit_button("🔍 Check Transaction")

if submitted:
    row = {
        "Time": transaction_time,
        **features,
        "Amount": amount
    }

    input_df = pd.DataFrame([row])

    try:
        prediction = int(model.predict(input_df)[0])

        if hasattr(model, "predict_proba"):
            fraud_probability = float(
                model.predict_proba(input_df)[0][1]
            )
        else:
            fraud_probability = None

        st.divider()

        if prediction == 1:
            st.error("🚨 Model Prediction: FRAUDULENT")
        else:
            st.success("✅ Model Prediction: GENUINE")

        if fraud_probability is not None:
            st.metric(
                "Model-estimated fraud probability",
                f"{fraud_probability:.2%}"
            )

        st.caption(
            "This is a machine-learning prediction, not a guarantee. "
            "It should not be used alone to approve or reject transactions."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
