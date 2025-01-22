import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('diamond_price_model.pkl')

# Streamlit application title
st.title("Diamond Price Prediction")

# User input for diamond features
carat = st.number_input("Carat", min_value=0.0, step=0.01)
cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
depth = st.number_input("Depth (%)", min_value=0.0, max_value=100.0, step=0.1)
table = st.number_input("Table (%)", min_value=0.0, max_value=100.0, step=0.1)
x = st.number_input("Length (mm)", min_value=0.0, step=0.1)
y = st.number_input("Width (mm)", min_value=0.0, step=0.1)
z = st.number_input("Depth (mm)", min_value=0.0, step=0.1)

# Button for prediction
if st.button("Predict Price"):
    # Prepare input data for prediction
    input_data = np.array([[carat, cut, color, clarity, depth, table, x, y, z]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display the prediction
    st.write(f"Predicted Price: ${prediction[0]:,.2f}")