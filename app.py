import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model once
model = load_model('your_model.h5')

st.title("Image Classifier (Input vector size 64)")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = img.resize((8, 8))  # Resize to 8x8
    img_array = image.img_to_array(img).flatten()  # Flatten to 1D
    img_array = img_array[:64]  # Use only first 64 values (or preprocess more cleanly if needed)
    img_array = img_array.reshape(1, 64)  # Shape: (1, 64)

    # Make prediction
    prediction = model.predict(img_array)

    st.write(f"Prediction: {float(prediction[0][0]):.4f}")
