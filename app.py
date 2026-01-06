import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model/churn_xgb.pkl")

st.title("📉 Customer Churn Prediction (XGBoost)")
st.write("Predict whether a customer will churn")

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.number_input("Tenure", 0, 72)
monthly = st.number_input("Monthly Charges", 0.0, 200.0)

contract = st.selectbox("Contract",
                        ["Month-to-month", "One year", "Two year"])

payment = st.selectbox("Payment Method",
                        ["Electronic check", "Mailed check",
                         "Bank transfer (automatic)",
                         "Credit card (automatic)"])

input_df = pd.DataFrame({
    "gender": ["Male"],
    "SeniorCitizen": [0],
    "Partner": ["Yes"],
    "Dependents": ["No"],
    "tenure": [tenure],
    "PhoneService": ["Yes"],
    "MultipleLines": ["No"],
    "InternetService": ["Fiber optic"],
    "OnlineSecurity": ["No"],
    "OnlineBackup": ["Yes"],
    "DeviceProtection": ["No"],
    "TechSupport": ["No"],
    "StreamingTV": ["Yes"],
    "StreamingMovies": ["No"],
    "Contract": [contract],
    "PaperlessBilling": ["Yes"],
    "PaymentMethod": [payment],
    "MonthlyCharges": [monthly],
    "TotalCharges": [monthly * tenure]
})



if st.button("Predict Churn"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"⚠️ High risk of churn ({prob:.2%})")
    else:
        st.success(f"✅ Low risk of churn ({1 - prob:.2%})")

