import streamlit as st
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "sleep_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

st.title("🛌 Sleep Disorder Prediction System")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 10, 100)
sleep_duration = st.number_input("Sleep Duration (hours)", 0.0, 15.0)
quality = st.slider("Quality of Sleep", 1, 10)
physical = st.slider("Physical Activity Level", 1, 10)
stress = st.slider("Stress Level", 1, 10)
bmi = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
heart_rate = st.number_input("Heart Rate", 40, 150)
steps = st.number_input("Daily Steps", 0, 30000)
systolic = st.number_input("Systolic BP", 80, 200)
diastolic = st.number_input("Diastolic BP", 50, 130)

gender = 0 if gender == "Male" else 1
bmi = {"Normal": 0, "Overweight": 1, "Obese": 2}[bmi]

input_data = np.array([[gender, age, sleep_duration, quality,
                        physical, stress, bmi, heart_rate,
                        steps, systolic, diastolic]])

input_scaled = scaler.transform(input_data)

if st.button("Predict"):
    result = model.predict(input_scaled)[0]

    if result == 0:
        st.success("✅ No Sleep Disorder")
    elif result == 1:
        st.warning("⚠️ Insomnia Detected")
    else:
        st.error("🚨 Sleep Apnea Detected")
