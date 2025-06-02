import os
import io
import unicodedata
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from fpdf import FPDF
from src.mlproject.predict_pipelines import PredictPipeline

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

st.set_page_config(page_title="🫀 Heart Risk & Diet AI", layout="wide")

# ------------------------- 🔐 Password Protection -------------------------
def check_password():
    def password_entered():
        if st.session_state["password"] == os.getenv("APP_PASSWORD"):
            st.session_state["authenticated"] = True
            del st.session_state["password"]
        else:
            st.session_state["authenticated"] = False

    if "authenticated" not in st.session_state:
        st.text_input("🔐 Enter App Password", type="password", on_change=password_entered, key="password")
        st.stop()
    elif not st.session_state["authenticated"]:
        st.text_input("🔐 Enter App Password", type="password", on_change=password_entered, key="password")
        st.error("❌ Incorrect password. Try again.")
        st.stop()

check_password()
# ------------------------- End Password Protection -------------------------

st.title("🫀 Heart Disease Predictor & Diet Assistant")

# Initialize session state keys
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "diet_plan_text" not in st.session_state:
    st.session_state.diet_plan_text = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

tab1 = st.container()

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
        st.session_state["predicted"] = True

    if st.session_state["predicted"]:
        st.markdown("---")
        if st.session_state["prediction"] == 1:
            st.error("⚠️ **High Risk of Heart Disease Detected!** Consult a cardiologist.")
        else:
            st.success("✅ **Low Risk of Heart Disease. Keep maintaining your health!**")

    if diet_btn:
        if not st.session_state["predicted"]:
            st.warning("⚠️ Please run the prediction first.")
        else:
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
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a certified medical dietitian."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000
                )
                st.session_state["diet_plan_text"] = response.choices[0].message.content

    if st.session_state["diet_plan_text"]:
        st.subheader("🥗 Recommended Diet Plan")
        formatted_diet = st.session_state["diet_plan_text"].replace("\n", "<br>")
        st.markdown(
            f"""
            <div style='background-color:#068f88;padding:15px;border-radius:10px;border:1px solid #ddd'>
            {formatted_diet}
            </div>
            """,
            unsafe_allow_html=True
        )

        def clean_text(text):
            return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

        cleaned_diet_text = clean_text(st.session_state["diet_plan_text"])

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

# ------------------------- SIDEBAR CHATBOT -------------------------
with st.sidebar:
    st.header("💬 Diet Chatbot")

    user_input = st.text_input("❓ Ask a diet-related question")

    if user_input:
        with st.spinner("🤖 Dietitian is typing..."):
            full_chat = [
                {"role": "system", "content": "You are a diet consultant bot named Healthy(B) made by Vivek. Always introduce yourself."},
            ]
            if st.session_state["diet_plan_text"]:
                full_chat.append({
                    "role": "user",
                    "content": f"This is my diet plan:\n{st.session_state['diet_plan_text']}"
                })
            full_chat.extend(st.session_state.chat_history)
            full_chat.append({"role": "user", "content": user_input})

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=full_chat,
                max_tokens=200
            )
            reply = response.choices[0].message.content
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("🧠 **Chat History**")
        for msg in reversed(st.session_state.chat_history):
            role_emoji = "👤" if msg["role"] == "user" else "🩺"
            st.markdown(f"{role_emoji} {msg['content']}")
