import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("model.h5")
class_labels = ['apple', 'kiwi', 'orange']  # Update this based on your model's classes

# Streamlit app title
st.title("🍎 Fruit Classifier App")

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png"])

# When the user uploads an image
if uploaded_file is not None:
    # Open and display the image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image to match the model's input
    img = img.resize((160, 160))  # Resize based on your model's expected input size
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction using the model
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)  # Get the class with the highest probability
    confidence = prediction[0][class_index]  # Get the confidence score

    # Display the prediction and confidence
    st.markdown(f"### Prediction: **{class_labels[class_index]}**")
    st.markdown(f"Confidence: `{confidence:.2f}`")

