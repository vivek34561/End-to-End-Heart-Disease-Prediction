import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from fpdf import FPDF
import io

from src.mlproject.predict_pipelines import PredictPipeline

# Load OpenAI key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="🫀 Heart Risk & Diet AI", layout="wide")
img = Image.open(r"C:\Users\vivek gupta\Desktop\Heart_disease_prediction\GettyImages-1155240409.jpg")

# Resize to specific width and height (e.g., 200x150)
img = img.resize((1300, 150))

st.title("🫀 Heart Disease Predictor & Diet Assistant")
st.image(img)

# # Optional infographic
# st.image(r"C:\Users\vivek gupta\Desktop\Heart_disease_prediction\GettyImages-1155240409.jpg", width=100)

# Tabs for layout
tab1, tab2 = st.tabs(["🔍 Risk & Diet Plan", "💬 Diet Chatbot"])

# ------------------------- TAB 1 -------------------------
with tab1:
    st.subheader("📋 Your Health Profile")

    with st.expander("🏠 Lifestyle & Demographics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("🎂 Age", 20, 90, 45)
            sex = st.radio("⚧️ Biological Sex", ["Male", "Female"])
        with col2:
            exang = st.radio("🏃 Do you get chest pain during exercise?", ["No", "Yes"])
            fbs = st.radio("🍬 Fasting blood sugar > 120 mg/dL?", ["No", "Yes"])

    with st.expander("💓 Vitals & Tests", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            trestbps = st.slider("🩺 Resting Blood Pressure (mm Hg)", 80, 200, 120)
            chol = st.slider("🧪 Cholesterol Level (mg/dL)", 100, 400, 220)
            thalach = st.slider("❤️ Max Heart Rate Achieved", 60, 210, 150)
        with col2:
            oldpeak = st.slider("📉 ST Depression (Exercise vs Rest)", 0.0, 6.0, 1.0, 0.1)
            restecg = st.selectbox("📈 ECG Results", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
            slope = st.selectbox("📐 Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])

    with st.expander("🧬 Medical History", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            cp = st.selectbox("💓 Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
        with col2:
            ca = st.selectbox("🩻 Number of Major Vessels Colored", [0, 1, 2, 3])
            thal = st.selectbox("🧬 Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    predict_btn = st.button("🚑 Predict Risk")
    diet_btn = st.button("🥗 Generate Diet Plan")

    # Shared model input dictionary
    model_input = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "cp": ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"].index(cp),
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 1 if fbs == "Yes" else 0,
        "restecg": ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"].index(restecg),
        "thalach": thalach,
        "exang": 1 if exang == "Yes" else 0,
        "oldpeak": oldpeak,
        "slope": ["Upsloping", "Flat", "Downsloping"].index(slope),
        "ca": ca,
        "thal": ["Normal", "Fixed Defect", "Reversible Defect"].index(thal),
    }

    if predict_btn:
     pipeline = PredictPipeline()
     prediction = pipeline.predict(model_input)
     st.session_state["prediction"] = prediction
     st.session_state["predicted"] = True  # ✅ Add a flag

     st.markdown("---")
     if prediction == 1:
        st.error("⚠️ **High Risk of Heart Disease Detected!** Consult a cardiologist.")
     else:
        st.success("✅ **Low Risk of Heart Disease. Keep maintaining your health!**")

    if diet_btn:
     if "predicted" not in st.session_state:
        st.warning("⚠️ Please run the prediction first.")
     else:
        prediction = st.session_state["prediction"]
        prompt = f"""
🧑‍⚕️ I’m a {age}-year-old {"male" if model_input['sex'] else "female"} with:
💉 BP: {trestbps} | 🧪 Cholesterol: {chol} | 🍬 Fasting Sugar: {"Yes" if model_input['fbs'] else "No"}
❤️ Max HR: {thalach} | 📉 ST Depression: {oldpeak} | 🧬 Thalassemia: {thal}

🍽️ Create a heart-healthy diet plan:
✅ Must include:
- 🌿 Essential nutrients (vitamins, minerals, macros)
- 🥗 Foods to eat & 🚫 avoid
- 🍳 Breakfast, 🍛 Lunch, 🍲 Dinner recipes
🎯 Tailor it to my condition & keep it practical.
"""


        with st.spinner("🍎 Generating personalized diet plan..."):
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a certified medical dietitian."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400  # increased to ensure detailed output
            )
            diet_text = response.choices[0].message.content

        st.subheader("🥗 Recommended Diet Plan")
        formatted_diet = diet_text.replace("\n", "<br>")
        st.markdown(
            f"""
            <div style='background-color:#f7f9fa;padding:15px;border-radius:10px;border:1px solid #ddd'>
            {formatted_diet}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Clean for PDF output
        import unicodedata

        def clean_text(text):
            return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

        cleaned_diet_text = clean_text(diet_text)

        # Generate and download PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)

        for line in cleaned_diet_text.split('\n'):
            pdf.multi_cell(0, 10, line)

        pdf_output = pdf.output(dest='S').encode('latin1')
        pdf_buffer = io.BytesIO(pdf_output)
        pdf_buffer.seek(0)

        st.download_button(
            label="📥 Download Diet Plan as PDF",
            data=pdf_buffer,
            file_name="heart_diet_plan.pdf",
            mime="application/pdf"
        )


# ------------------------- TAB 2 -------------------------
with tab2:
    st.subheader("💬 Ask Your Diet Question")
    user_query = st.text_input("❓ Your diet question")

    if user_query:
        chat_prompt = f"You are a helpful heart nutrition expert. Question:\n\n{user_query}"
        with st.spinner("💡 Thinking..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional heart-health dietitian."},
                    {"role": "user", "content": chat_prompt}
                ],
                max_tokens=400,
                temperature=0.7  # fixed typo
            )
        answer = response.choices[0].message.content
        st.markdown("✅ **Expert Answer:**")
        st.markdown(f"<div style='background-color:#f7f9fa;padding:15px;border-radius:10px;border:1px solid #ddd'>{answer.replace(chr(10),'<br>')}</div>", unsafe_allow_html=True)

