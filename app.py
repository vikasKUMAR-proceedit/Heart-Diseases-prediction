import streamlit as st
import joblib
import numpy as np

# Page setup - modern look
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('heart_disease.pkl')

model = load_model()

# Header
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>🫀 Heart Disease Risk Predictor</h1>
    <p style='text-align: center; font-size: 1.2rem; color: #666;'>
    AI-powered prediction using Random Forest on UCI Heart Disease Dataset<br>
    <strong style='color: #FF6B6B;'>⚠️ For educational purposes only — not a substitute for medical advice!</strong>
    </p>
    <hr style='border-color: #444;'>
""", unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.image("https://img.icons8.com/emoji/100/000000/red-heart.png")
    st.header("📋 Feature Ranges (from dataset)")
    st.write("""
    - **Age**: 29–77 years
    - **Cholesterol**: 126–564 mg/dl
    - **Max Heart Rate**: 71–202 bpm
    - **Resting BP**: 94–200 mm Hg
    - **ST Depression**: 0.0–6.2
    """)
    st.caption("Model: RandomForestClassifier | Accuracy ~83-85%")

# Input columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal & Symptoms")
    age = st.slider("Age (years)", 18, 100, 54, help="Patient age in years")
    sex = st.selectbox("Sex", options=[1, 0], index=0, format_func=lambda x: "Male" if x == 1 else "Female")  # Default: Male
    cp = st.selectbox("Chest Pain Type", options=[0,1,2,3], index=0,
                      format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])  # Default: Typical Angina

    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 94, 200, 130)
    chol = st.slider("Cholesterol (mg/dl)", 126, 564, 240)

with col2:
    st.subheader("🩺 Clinical Measurements")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[1, 0], index=1, format_func=lambda x: "Yes" if x == 1 else "No")  # Default: No (more common)
    restecg = st.selectbox("Resting ECG", options=[0,1,2], index=0,
                           format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x])  # Default: Normal
    thalach = st.slider("Max Heart Rate Achieved", 71, 202, 150)

    exang = st.selectbox("Exercise-Induced Angina", options=[0, 1], index=0, format_func=lambda x: "No" if x == 0 else "Yes")  # Default: No
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0, step=0.1)
    slope = st.selectbox("ST Segment Slope", options=[0,1,2], index=1,
                         format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])  # Default: Flat (common)
    ca = st.slider("Major Vessels (fluoroscopy)", 0, 4, 0)

    # FIXED: thal values are 1,2,3 in your dataset!
    thal = st.selectbox("Thalassemia", options=[1, 2, 3], index=0,
                        format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[x])  # Default: Normal

# Prediction button
if st.button("🔬 Predict Risk", type="primary", use_container_width=True):
    # Create input array in correct column order
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]  # Probability of disease (class 1)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Result display
    col_a, col_b, col_c = st.columns([1, 3, 1])
    with col_b:
        if prediction == 1:
            st.markdown(f"""
                <h2 style='text-align: center; color: #FF4444;'>⚠️ High Risk Detected</h2>
                <p style='text-align: center; font-size: 1.4rem;'>
                The model predicts presence of heart disease.
                </p>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <h2 style='text-align: center; color: #44FF44;'>✅ Low Risk</h2>
                <p style='text-align: center; font-size: 1.4rem;'>
                The model predicts no heart disease.
                </p>
            """, unsafe_allow_html=True)

        # Probability gauge
        st.markdown(f"<h3 style='text-align: center;'>Risk Probability: {probability:.1%}</h3>", unsafe_allow_html=True)
        st.progress(probability)

        if probability > 0.75:
            st.error("🚨 Very high risk — strongly recommend seeing a cardiologist.")
        elif probability > 0.5:
            st.warning("🟠 Moderate to high risk — consider lifestyle changes and medical checkup.")
        else:
            st.success("🟢 Low risk — maintain healthy habits!")

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center; color: #888; font-size: 0.9rem;'>
    Built with ❤️ using Streamlit • Model: Random Forest • Dataset: UCI Heart Disease
    </p>
""", unsafe_allow_html=True)

