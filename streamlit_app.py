import streamlit as st
from src.mlproject.predict_pipelines import PredictPipeline

st.set_page_config(page_title="🫀 Heart Disease Classifier", layout="wide")
st.title("🏥 Heart Disease Prediction App")

# Layout in 3 columns
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=43.0)
    gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=300.0, value=177.0)
    cholesterol = st.number_input("Cholesterol Level", min_value=0.0, max_value=400.0, value=175.0)
    exercise = st.selectbox("Exercise Habits", ["High", "Medium", "Low"], index=0)
    smoking = st.selectbox("Smoking", ["Yes", "No"], index=1)
    family_history = st.selectbox("Family Heart Disease", ["Yes", "No"], index=0)

with col2:
    diabetes = st.selectbox("Diabetes", ["Yes", "No"], index=1)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=30.88652065200732)
    high_bp = st.selectbox("High Blood Pressure", ["Yes", "No"], index=0)
    low_hdl = st.selectbox("Low HDL Cholesterol", ["Yes", "No"], index=1)
    high_ldl = st.selectbox("High LDL Cholesterol", ["Yes", "No"], index=0)
    alcohol = st.selectbox("Alcohol Consumption", ["Low", "Medium", "High"], index=2)
    stress = st.selectbox("Stress Level", ["Low", "Medium", "High"], index=1)

with col3:
    sleep = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=9.262954945724116)
    sugar = st.selectbox("Sugar Consumption", ["Low", "Medium", "High"], index=2)
    triglyceride = st.number_input("Triglyceride Level", min_value=0.0, max_value=1000.0, value=253.0)
    fasting_sugar = st.number_input("Fasting Blood Sugar", min_value=0.0, max_value=300.0, value=131.0)
    crp = st.number_input("CRP Level", min_value=0.0, max_value=100.0, value=4.106257243372669)
    homocysteine = st.number_input("Homocysteine Level", min_value=0.0, max_value=100.0, value=10.582307997062776)

# Input dictionary
input_data = {
    "Age": age,
    "Gender": gender,
    "Blood Pressure": blood_pressure,
    "Cholesterol Level": cholesterol,
    "Exercise Habits": exercise,
    "Smoking": smoking,
    "Family Heart Disease": family_history,
    "Diabetes": diabetes,
    "BMI": bmi,
    "High Blood Pressure": high_bp,
    "Low HDL Cholesterol": low_hdl,
    "High LDL Cholesterol": high_ldl,
    "Alcohol Consumption": alcohol,
    "Stress Level": stress,
    "Sleep Hours": sleep,
    "Sugar Consumption": sugar,
    "Triglyceride Level": triglyceride,
    "Fasting Blood Sugar": fasting_sugar,
    "CRP Level": crp,
    "Homocysteine Level": homocysteine
}

# Predict button
if st.button("🔍 Predict Heart Disease Status"):
    pipeline = PredictPipeline()
    prediction = pipeline.predict(input_data)

    if prediction == 1 or prediction == "Yes":
        st.error("⚠️ High Risk of Heart Disease Detected!")
    else:
        st.success("✅ Low Risk of Heart Disease.")
