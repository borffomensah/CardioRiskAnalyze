import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained SVM model
with open('best_svm_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit App
st.title("❤️ Heart Disease Detection App")
st.markdown("<h5 style='text-align: center;'>Early Detection, Stronger Protection.</h5>", unsafe_allow_html=True)

# Move the input form to the sidebar
st.sidebar.header("Patient Information")

# First row
age = st.sidebar.slider('Age', 20, 100, 30)
sex = st.sidebar.radio('Sex', ('Male', 'Female'))
cp = st.sidebar.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
trestbps = st.sidebar.slider('Resting BP (mm Hg)', 80, 200, 120)

# Second row
chol = st.sidebar.slider('Cholesterol (mg/dl)', 100, 600, 200)
fbs = st.sidebar.radio('Fasting Blood Sugar > 120', ('Yes', 'No'))
restecg = st.sidebar.selectbox('Resting ECG', ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy'])
thalach = st.sidebar.slider('Max Heart Rate', 70, 220, 150)

# Third row
exang = st.sidebar.radio('Exercise Induced Angina', ('Yes', 'No'))
oldpeak = st.sidebar.slider('Oldpeak', 0.0, 6.0, 1.0, step=0.1)
slope = st.sidebar.selectbox('Slope', ['Upsloping', 'Flat', 'Downsloping'])
ca = st.sidebar.slider('Major Vessels (0-3)', 0, 3, 0)

# Fourth row (only one input)
thal = st.sidebar.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

# --- Data Preprocessing ---
sex = 1 if sex == 'Male' else 0
cp = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(cp)
fbs = 1 if fbs == 'Yes' else 0
restecg = ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy'].index(restecg)
exang = 1 if exang == 'Yes' else 0
slope = ['Upsloping', 'Flat', 'Downsloping'].index(slope)
thal = ['Normal', 'Fixed Defect', 'Reversible Defect'].index(thal) + 3

# --- Prediction ---
st.markdown("---")
if st.button('🔍 Predict'):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                            exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success('✅ No Heart Disease Detected.')
    else:
        st.error('⚠️ Heart Disease Detected. Please consult a healthcare provider.')

