# 🫀 Heart Disease Predictor & AI Diet Assistant

An AI-powered Streamlit web application that predicts the risk of heart disease using ML models and generates a personalized, heart-healthy diet plan using OpenAI's GPT-4o.

---

## 📌 Features

### 🔍 Heart Disease Prediction
- Predicts risk based on user inputs like age, cholesterol, chest pain type, etc.
- Uses advanced ML models: RandomForest, XGBoost, CatBoost.
- Selects and saves the best model based on evaluation metrics.
- MLflow integrated for experiment tracking and model registry.

### 🥗 AI-Powered Diet Generator
- Personalized heart-healthy diet plan using OpenAI GPT-4o.
- Considers user’s weight, height, age, gender, lifestyle, and food preference.
- Outputs meals for breakfast, lunch, dinner + what to avoid.
- One-click **PDF download** of diet plan.

### 💬 Diet Chatbot Assistant
- Sidebar chatbot answers user’s dietary questions.
- Powered by OpenAI GPT-4o.
- Contextual conversation experience.

### 📸 Screenshots

| Heart Disease Predictor | Personalized Diet Plan |
|-------------------------|-------------------------|
| ![Predictor](screenshots/predictor.png) | ![Diet Plan](screenshots/diet.png) |

---

## 🌐 Live App

🔗 [Access the App](https://your-deployed-app-link.com)

---

## ⚙️ Technologies Used

- **Python 3.9+**
- **Scikit-learn**, **XGBoost**, **CatBoost**
- **MLflow** for model tracking
- **Streamlit** for frontend
- **OpenAI GPT-4o** for diet generation & chatbot
- **MySQL** for patient data
- **FPDF** for diet plan PDF download
- **dotenv** for environment variable management


