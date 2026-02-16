import numpy as np
import pandas as pd
import pickle
import streamlit as st
import os
import joblib


model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("standard_scaler.pkl")

st.title("💓 Heart Attack Prediction App")

st.write("Please fill in the patient's medical information:")

age = st.number_input("Age", min_value=0, max_value=120, value=50)
gender = st.selectbox("Gender", ("Male", "Female"))
heart_rate = st.number_input("Heart Rate", min_value=0, value=70)
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=0, value=120)
diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=0, value=80)
blood_sugar = st.number_input("Blood Sugar", min_value=0.0, value=120.0)
ck_mb = st.number_input("CK-MB", min_value=0.0, value=1.0)
troponin = st.number_input("Troponin", min_value=0.0, value=0.01)

gender_encoded = 1 if gender == "Male" else 0
input_data = np.array([[age, gender_encoded, heart_rate, systolic_bp, diastolic_bp, blood_sugar, ck_mb, troponin]])
input_data_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    result = label_encoder.inverse_transform(prediction)[0]

    if result == "positive":
        st.error("⚠️ High Risk of Heart Attack Detected!")
    else:
        st.success("✅ Low Risk of Heart Attack")