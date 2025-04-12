import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model("model.h5")

st.title("Fruit Classifier (with Numeric Inputs)")

st.write("Enter the 8 features below:")

# Create 8 number inputs
features = []
for i in range(8):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    features.append(value)

# Convert to NumPy array and reshape
features_array = np.array([features])  # shape (1, 8)

if st.button("Predict"):
    prediction = model.predict(features_array)
    st.write("Prediction:", prediction[0][0])
