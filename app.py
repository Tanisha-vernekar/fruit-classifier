import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

st.title("Simple Image Classifier")

# Load model
model = load_model("your_model.h5")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = img.resize((8, 8))  # Resize to 8x8 pixels
    img_array = image.img_to_array(img).flatten()[:64]  # Get first 64 features
    img_array = img_array.reshape(1, 64)  # Shape to (1, 64)

    # Predict
    prediction = model.predict(img_array)
    st.write(f"Prediction: {float(prediction[0][0]):.4f}")
