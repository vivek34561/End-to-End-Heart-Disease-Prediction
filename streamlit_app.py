import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open("artifacts/model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Heart Disease Risk Prediction")

# Input form
with st.form("heart_disease_form"):
    age = st.number_input("Age", 0, 120)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bp = st.number_input("Blood Pressure", 0.0)
    chol = st.number_input("Cholesterol Level", 0.0)
    exercise = st.selectbox("Exercise Habits", ["Low", "Medium", "High"])
    smoking = st.selectbox("Smoking", ["Yes", "No"])
    family_history = st.selectbox("Family Heart Disease", ["Yes", "No"])
    diabetes = st.selectbox("Diabetes", ["Yes", "No"])
    bmi = st.number_input("BMI", 0.0)
    hbp = st.selectbox("High Blood Pressure", ["Yes", "No"])
    low_hdl = st.selectbox("Low HDL Cholesterol", ["Yes", "No"])
    high_ldl = st.selectbox("High LDL Cholesterol", ["Yes", "No"])
    alcohol = st.selectbox("Alcohol Consumption", ["Low", "Medium", "High"])
    stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
    sleep = st.number_input("Sleep Hours", 0.0)
    sugar = st.selectbox("Sugar Consumption", ["Low", "Medium", "High"])
    triglyceride = st.number_input("Triglyceride Level", 0.0)
    fbs = st.number_input("Fasting Blood Sugar", 0.0)
    crp = st.number_input("CRP Level", 0.0)
    homocysteine = st.number_input("Homocysteine Level", 0.0)

    submitted = st.form_submit_button("Predict Heart Disease Risk")

# Mapping for categorical variables
def encode_input(df):
    mappings = {
        "Gender": {"Male": 1, "Female": 0},
        "Exercise Habits": {"Low": 0, "Medium": 1, "High": 2},
        "Smoking": {"No": 0, "Yes": 1},
        "Family Heart Disease": {"No": 0, "Yes": 1},
        "Diabetes": {"No": 0, "Yes": 1},
        "High Blood Pressure": {"No": 0, "Yes": 1},
        "Low HDL Cholesterol": {"No": 0, "Yes": 1},
        "High LDL Cholesterol": {"No": 0, "Yes": 1},
        "Alcohol Consumption": {"Low": 0, "Medium": 1, "High": 2},
        "Stress Level": {"Low": 0, "Medium": 1, "High": 2},
        "Sugar Consumption": {"Low": 0, "Medium": 1, "High": 2},
    }

    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping)

    return df

# Prediction
if submitted:
    input_data = {
        "Age": age,
        "Gender": gender,
        "Blood Pressure": bp,
        "Cholesterol Level": chol,
        "Exercise Habits": exercise,
        "Smoking": smoking,
        "Family Heart Disease": family_history,
        "Diabetes": diabetes,
        "BMI": bmi,
        "High Blood Pressure": hbp,
        "Low HDL Cholesterol": low_hdl,
        "High LDL Cholesterol": high_ldl,
        "Alcohol Consumption": alcohol,
        "Stress Level": stress,
        "Sleep Hours": sleep,
        "Sugar Consumption": sugar,
        "Triglyceride Level": triglyceride,
        "Fasting Blood Sugar": fbs,
        "CRP Level": crp,
        "Homocysteine Level": homocysteine,
    }

    df = pd.DataFrame([input_data])
    df = encode_input(df)

    prediction = model.predict(df)[0]

    st.subheader("Prediction:")
    if prediction == 1 or prediction == "Yes":
        st.error("🚨 High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
