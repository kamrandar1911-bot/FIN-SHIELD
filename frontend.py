
import streamlit as st
import pandas as pd
import joblib
from datetime import date, time

# Page settings
st.set_page_config(
    page_title="FIN-SHIELD",
    page_icon="🛡️",
    layout="centered"
)

# Load the complete trained Pipeline
@st.cache_resource
def load_model():
    return joblib.load("fin_shield_model_new.joblib")

try:
    model = load_model()
except Exception as e:
    st.error("Model load nahi hua.")
    st.code(str(e))
    st.stop()

# Website heading
st.title("🛡️ FIN-SHIELD")
st.subheader("Credit Card Fraud Detection")
st.write(
    "Enter transaction details and check the model's prediction."
)

st.divider()

# Input form
with st.form("fraud_form"):
    st.subheader("Transaction Details")

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.01,
        value=1000.0,
        step=50.0
    )

    merchant_id = st.number_input(
        "Merchant ID",
        min_value=0,
        value=100,
        step=1
    )

    transaction_type = st.selectbox(
        "Transaction Type",
        ["purchase", "refund"]
    )

    location = st.selectbox(
        "Location",
        [
            "Chicago", "San Diego", "Dallas",
            "San Antonio", "New York", "Houston",
            "Phoenix", "Los Angeles", "Philadelphia",
            "San Jose"
        ]
    )

    transaction_date = st.date_input(
        "Transaction Date",
        value=date.today()
    )

    transaction_time = st.time_input(
        "Transaction Time",
        value=time(12, 0)
    )

    submitted = st.form_submit_button(
        "Check Transaction",
        use_container_width=True
    )

# Prediction
if submitted:
    dt = pd.Timestamp.combine(
        transaction_date,
        transaction_time
    )

    input_data = pd.DataFrame([{
        "Amount": amount,
        "MerchantID": str(int(merchant_id)),
        "TransactionType": transaction_type,
        "Location": location,
        "Hour": dt.hour,
        "DayOfWeek": dt.dayofweek,
        "Month": dt.month
    }])

    try:
        prediction = int(model.predict(input_data)[0])
        fraud_probability = float(
            model.predict_proba(input_data)[0][1]
        )

        st.divider()
        st.subheader("Prediction Result")

        st.metric(
            "Estimated Fraud Probability",
            f"{fraud_probability:.2%}"
        )

        if prediction == 1:
            st.error(
                "⚠️ POTENTIAL FRAUD — Review this transaction."
            )
        else:
            st.success(
                "✅ LIKELY GENUINE — The model did not flag it."
            )

        st.caption(
            "Experimental model only. This prediction is not "
            "proof that a transaction is genuine or fraudulent."
        )

    except Exception as e:
        st.error("Prediction mein error aaya.")
        st.code(str(e))

st.divider()
st.caption("FIN-SHIELD | AI-Based Fraud Detection")
