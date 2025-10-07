import streamlit as st
import numpy as np
import joblib
import base64 # New import

# --- FUNCTION TO ADD A BACKGROUND IMAGE ---
# The function reads a local image, encodes it, and applies it as the background.
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# --- LOAD THE TRAINED MODEL ---
@st.cache_resource
def load_model():
    model = joblib.load('heart_disease_model.joblib')
    return model

model = load_model()

# --- WEB APP INTERFACE ---

# Call the function to add the background image
# MAKE SURE 'background.jpg' is the correct name of your image file
add_bg_from_local('background.jpg') 

# Set the title of the web app
st.title('❤️ Heart Disease Prediction App')
# ... (the rest of your code remains exactly the same)
st.write("This app predicts whether a person has heart disease based on their medical attributes.")

# Create input fields for the user in two columns
st.header("Please enter the patient's details:")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=1, max_value=120, value=50)
    sex = st.selectbox('Sex', (0, 1), format_func=lambda x: 'Female' if x == 0 else 'Male')
    cp = st.selectbox('Chest Pain Type (cp)', (0, 1, 2, 3))
    trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=80, max_value=200, value=120)
    chol = st.number_input('Serum Cholestoral in mg/dl (chol)', min_value=100, max_value=600, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', (0, 1))

with col2:
    restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', (0, 1, 2))
    thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60, max_value=220, value=150)
    exang = st.selectbox('Exercise Induced Angina (exang)', (0, 1))
    oldpeak = st.number_input('ST depression induced by exercise relative to rest (oldpeak)', min_value=0.0, max_value=7.0, value=1.0, step=0.1)
    slope = st.selectbox('The slope of the peak exercise ST segment (slope)', (0, 1, 2))
    ca = st.selectbox('Number of major vessels (0-3) colored by flourosopy (ca)', (0, 1, 2, 3, 4))
    thal = st.selectbox('Thal', (0, 1, 2, 3))


# --- PREDICTION LOGIC ---

# Create a button to trigger the prediction
if st.button('Predict Heart Disease', type="primary"):
    # Collect the input data into a single list or array
    # The order MUST match the order of columns in the training data
    input_data = np.array([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]])

    # Make a prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Display the result
    st.subheader('Prediction Result:')

    if prediction[0] == 0:
        st.success('**Healthy Heart**')
        st.write(f"Confidence: {prediction_proba[0][0]*100:.2f}%")
        st.balloons()
    else:
        st.error('**Heart Disease Detected**')
        st.write(f"Confidence: {prediction_proba[0][1]*100:.2f}%")
        st.warning("Please consult a doctor for further evaluation.")


st.sidebar.info("This is a web application for predicting heart disease using a machine learning model. Created for educational purposes.")
