import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('final_model.pkl')      # You must have saved this earlier
         

# Streamlit app layout
st.title("🍷 Wine Quality Prediction App")
st.write("Enter the wine features below to predict whether the wine is *Good (1)* or *Bad (0)*.")

# Input fields
fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 8.0)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.5)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.slider("Residual Sugar", 0.5, 15.0, 2.5)
chlorides = st.slider("Chlorides", 0.01, 0.2, 0.05)
free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1, 70, 15)
total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6, 200, 50)
density = st.slider("Density", 0.9900, 1.0050, 0.9950)
pH = st.slider("pH", 2.8, 4.0, 3.2)
sulphates = st.slider("Sulphates", 0.2, 1.5, 0.6)
alcohol = st.slider("Alcohol", 8.0, 14.9, 10.0)

# Combine inputs into array
input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                        residual_sugar, chlorides, free_sulfur_dioxide,
                        total_sulfur_dioxide, density, pH, sulphates, alcohol]])



# Predict button
if st.button("Predict Wine Quality"):
    prediction = model.predict(scaled_data)[0]
    result = "🍷 Good Quality Wine (1)" if prediction == 1 else "⚠ Bad Quality Wine (0)"
    st.success(f"Prediction: {result}")
