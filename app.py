import streamlit as st
import numpy as np
import pickle

# Load trained model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("🍷 Wine Quality Predictor")
st.write("Enter the wine's chemical properties to predict its quality.")

# Feature input fields
features = {
    'fixed acidity': st.number_input('Fixed Acidity', value=7.0),
    'volatile acidity': st.number_input('Volatile Acidity', value=0.7),
    'citric acid': st.number_input('Citric Acid', value=0.0),
    'residual sugar': st.number_input('Residual Sugar', value=1.9),
    'chlorides': st.number_input('Chlorides', value=0.076),
    'free sulfur dioxide': st.number_input('Free Sulfur Dioxide', value=11.0),
    'total sulfur dioxide': st.number_input('Total Sulfur Dioxide', value=34.0),
    'density': st.number_input('Density', value=0.9978),
    'pH': st.number_input('pH', value=3.51),
    'sulphates': st.number_input('Sulphates', value=0.56),
    'alcohol': st.number_input('Alcohol (%)', value=9.4)
}

# Predict button
if st.button("Predict Quality"):
    input_data = np.array([list(features.values())]).reshape(1, -1)
    
    # Apply the same scaling as used during training
    input_data_scaled = scaler.transform(input_data)
    
    # Make the prediction with the scaled data
    prediction = model.predict(input_data_scaled)
    
    # Map prediction to quality label
    quality_label = {0: 'Low ((3-5)/10)', 1: 'Medium (6/10)', 2: 'High ((7-10)/10)'}
    quality_category = quality_label.get(prediction[0], "Unknown")

    st.success(f"Predicted Wine Quality: {quality_category}")
