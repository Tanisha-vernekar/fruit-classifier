import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model("model.h5")

# Define the class labels (same order as training)
class_labels = ["Apple", "Kiwi", "Orange"]

# Set the image size as used in training (IMPORTANT!)
IMAGE_SIZE = (200, 160)  # (height, width)

# Streamlit UI
st.title("🍎🥝🍊 Fruit Image Classifier")
st.write("Upload an image of a fruit (Apple, Kiwi, or Orange) and the model will predict it.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # --- PREPROCESSING ---
    img = image.resize(IMAGE_SIZE)          # Resize to (200, 160)
    img = img_to_array(img)                 # Convert to array (200, 160, 3)
    img = img / 255.0                       # Normalize if model trained with rescale=1./255
    img = np.expand_dims(img, axis=0)      # Add batch dimension: (1, 200, 160, 3)

    # Show debug info (optional)
    st.write("📐 Model expects:", model.input_shape)
    st.write("📐 Image input shape:", img.shape)

    # --- PREDICTION ---
    prediction = model.predict(img)
    predicted_class = class_labels[np.argmax(prediction)]

    # --- OUTPUT ---
    st.success(f"✅ Prediction: **{predicted_class}**")
