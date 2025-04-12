import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model("model.h5")

st.title("Fruit Classifier (with Numeric Inputs)")

st.write("Enter the 3 features below:")

# Custom feature names
feature_names = ["Size", "Color", "Shape",]

# Create 8 number inputs based on the feature names
features = []
for i in range(3):
    value = st.number_input(feature_names[i], value=0.0)
    features.append(value)

# Convert to NumPy array and reshape
features_array = np.array([features])  # shape (1, 8)

if st.button("Predict"):
    # Make the prediction
    prediction = model.predict(features_array)
    
    # Get the predicted class (which fruit has the highest probability)
    predicted_class_index = np.argmax(prediction)  # index of the highest probability
    
    # Map the index to fruit names
    fruit_classes = ["Kiwi", "Apple", "Orange"]
    predicted_class = fruit_classes[predicted_class_index]
    
    st.write("Prediction:", predicted_class)
