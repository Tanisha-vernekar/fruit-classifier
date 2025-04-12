import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model("model.h5")

# Define the fruit labels (same order as your training)
class_labels = ["Apple", "Kiwi", "Orange"]

# Set the image size your model expects (change if needed)
IMAGE_SIZE = (200, 160)

# App title
st.title("🍎🥝🍊 Fruit Classifier")
st.write("Upload an image of a fruit (Apple, Kiwi, or Orange), and the model will predict which fruit it is.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img = image.resize(IMAGE_SIZE)
    img = img_to_array(img)
    img = img / 255.0  # Normalize if model was trained with rescale=1./255
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img)
    predicted_class = class_labels[np.argmax(prediction)]

    # Show result
    st.success(f"Prediction: **{predicted_class}**")
