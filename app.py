import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Sleep Disorder Prediction",
    page_icon="🌙",
    layout="centered"
)


st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
h1, h2, h3, h4, p, label {
    color: #e5e7eb !important;
}
div.stButton > button {
    background-color: #6366f1;
    color: white;
    font-size: 18px;
    padding: 12px 28px;
    border-radius: 14px;
}
div.stButton > button:hover {
    background-color: #4f46e5;
}
.card {
    background-color:#1f2937;
    padding:20px;
    border-radius:18px;
    margin-bottom:15px;
}
</style>
""", unsafe_allow_html=True)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "sleep_model.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))


st.markdown("""
<div style="
background: linear-gradient(135deg,#6366f1,#8b5cf6);
padding: 35px;
border-radius: 22px;
text-align: center;
box-shadow: 0px 10px 30px rgba(0,0,0,0.3);
">
<h1>🛌 Sleep Disorder Prediction System</h1>
<p style="font-size:18px;">
AI-based health dashboard to analyze sleep patterns and lifestyle factors
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


with st.expander("👤 Personal Information", expanded=True):
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 10, 100)

with st.expander("😴 Sleep Details", expanded=True):
    sleep_duration = st.number_input("Sleep Duration (hours)", 0.0, 15.0)
    sleep_quality_text = st.text_area(
        "Describe your sleep quality 📝",
        placeholder="e.g. frequent waking, difficulty sleeping, nightmares, deep sleep",
        height=120
    )

with st.expander("🏃 Lifestyle Factors"):
    physical = st.slider("Physical Activity Level", 1, 10)
    stress = st.slider("Stress Level", 1, 10)

with st.expander("❤️ Health Parameters"):
    bmi = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
    heart_rate = st.number_input("Heart Rate (bpm)", 40, 150)
    steps = st.number_input("Daily Steps", 0, 30000)

    col1, col2 = st.columns(2)
    with col1:
        systolic = st.number_input("Systolic BP", 80, 200)
    with col2:
        diastolic = st.number_input("Diastolic BP", 50, 130)


gender = 0 if gender == "Male" else 1
bmi = {"Normal": 0, "Overweight": 1, "Obese": 2}[bmi]


def quality_from_text(text):
    text = text.lower()

    if any(word in text for word in ["excellent", "peaceful", "deep", "restful"]):
        return 9
    elif any(word in text for word in ["good", "fine", "okay"]):
        return 7
    elif any(word in text for word in ["waking", "disturbed", "light sleep", "restless"]):
        return 5
    elif any(word in text for word in ["insomnia", "no sleep", "nightmare", "anxiety", "stress"]):
        return 3
    else:
        return 6  # default

quality = quality_from_text(sleep_quality_text)

st.markdown(
    f"""
    <div class="card" style="text-align:center;">
    🧠 <b>Derived Sleep Quality Score:</b> {quality}/10
    </div>
    """,
    unsafe_allow_html=True
)


input_data = np.array([[gender, age, sleep_duration, quality,
                        physical, stress, bmi, heart_rate,
                        steps, systolic, diastolic]])

input_scaled = scaler.transform(input_data)


st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict = st.button("🔍 Predict Sleep Disorder")


if predict:
    result = model.predict(input_scaled)[0]

    if result == 0:
        st.markdown("""
        <div class="card" style="background:#064e3b;">
        <h3>✅ No Sleep Disorder Detected</h3>
        <p>Your sleep pattern appears healthy.</p>
        </div>
        """, unsafe_allow_html=True)

    elif result == 1:
        st.markdown("""
        <div class="card" style="background:#78350f;">
        <h3>⚠️ Insomnia Detected</h3>
        <p>Consider improving sleep routine and stress management.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="card" style="background:#7f1d1d;">
        <h3>🚨 Sleep Apnea Detected</h3>
        <p>Medical consultation is recommended.</p>
        </div>
        """, unsafe_allow_html=True)


st.markdown("""
<div style="text-align:center; opacity:0.6; margin-top:30px;">
🌙 Final Year Project | Sleep Disorder Prediction using Machine Learning
</div>
""", unsafe_allow_html=True)
