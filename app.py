import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("random_forest_model1.sav")
scaler = joblib.load("scaler1.sav")

# App configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="logo.png",
    layout="centered"
)

# Sidebar navigation
page = st.sidebar.radio("Navigate", ["💓 Heart Risk Predictor", "🚨 Emergency Checker", "ℹ️ About"])

# HEART RISK PREDICTOR PAGE
if page == "💓 Heart Risk Predictor":
    st.image("logo.png", width=120)
    st.title("Heart Disease Risk Prediction App")

    st.markdown("Fill in the details below to assess your potential heart disease risk.")

    # Input sections
    st.subheader("🧍 Patient Information")
    age = st.slider("Age", 20, 100, 50)
    sex = st.radio("Sex", ["Male", "Female"])
    sex = 1 if sex == "Male" else 0

    st.subheader("❤️ Heart Health Indicators")
    cp = st.selectbox("Chest Pain Type", ["Typical Angina (0)", "Atypical Angina (1)", "Non-anginal (2)", "Asymptomatic (3)"])
    cp = int(cp[-2]) if cp[-2].isdigit() else int(cp[-1])
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.radio("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
    oldpeak = st.slider("ST Depression (0.0–6.0)", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

    # Prepare input data
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]],
                              columns=['Age', 'Sex', 'Chest_Pain_Type', 'Resting_BP',
                                       'Serum_Cholesterol', 'Fasting_Blood_Sugar',
                                       'Resting_ECG', 'Max_Heart_Rate', 'Exercise_Angina',
                                       'ST_Depression', 'Slope_ST_Segment', 'Major_Vessels', 'Thalassemia'])

    # Scale numerical columns
    cols_to_scale = ['Age', 'Resting_BP', 'Serum_Cholesterol', 'Max_Heart_Rate', 'ST_Depression']
    input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])

    # Prediction button
    if st.button("🔍 Predict Risk"):
        try:
            prediction = model.predict(input_data)[0]

            st.subheader("🩺 Prediction Result")
            if prediction == 1:
                st.error("🔴 **Heart Disease Detected**")
                st.warning("⚠️ Please consult a healthcare provider for a full diagnosis and treatment plan.")

                # Recommendations
                st.subheader("📋 Recommendations")
                st.info("""
                - Schedule an appointment with a cardiologist as soon as possible.  
                - Adopt a heart-healthy diet (low salt, low fat, high fiber).  
                - Engage in light physical activity if approved by your doctor.  
                - Avoid smoking and manage stress.  
                - Monitor blood pressure and cholesterol regularly.
                """)
            else:
                st.success("🟢 **No Heart Disease Detected**")
                st.info("💪 Great job! Keep maintaining your heart health with these tips:")
                st.subheader("📋 Recommendations")
                st.info("""
                - Continue regular exercise (at least 30 minutes daily).  
                - Eat a balanced diet rich in fruits and vegetables.  
                - Keep track of blood pressure and cholesterol.  
                - Avoid smoking and manage stress effectively.
                """)
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# -------------------------------
# EMERGENCY CHECKER PAGE
# -------------------------------
elif page == "🚨 Emergency Checker":
    st.image("logo.png", width=120)
    st.title("🚨 Emergency Symptom Checker")

    st.markdown("Check any symptoms you are experiencing:")

    chest_pain = st.checkbox("Chest Pain")
    breath = st.checkbox("Shortness of Breath")
    dizziness = st.checkbox("Dizziness or Fatigue")
    sweating = st.checkbox("Unusual Sweating")
    nausea = st.checkbox("Nausea or Lightheadedness")

    if st.button("Check Symptoms"):
        if chest_pain and breath:
            st.error("⚠️ **Seek emergency care immediately!** These may indicate a serious heart condition.")
        elif chest_pain or dizziness or nausea:
            st.warning("⚠️ **Consult a doctor soon.** Some symptoms may require medical attention.")
        else:
            st.success("✅ No urgent symptoms detected. Maintain regular health check-ups.")

# ABOUT PAGE
elif page == "ℹ️ About":
    st.image("logo.png", width=120)
    st.title("ℹ️ About This App")
    st.write("""
    This Heart Disease Prediction App uses a pre-trained **Random Forest model** to estimate
    the likelihood of heart disease based on medical inputs.

    **Developed by:** Team Heart Bits  
    **Model:** Random Forest Classifier  
    **Purpose:** Early risk detection and awareness for heart health.  
    """)

    st.markdown("💡 *Disclaimer: This app is for educational and informational purposes only. Always consult a medical professional for diagnosis or treatment.*")
