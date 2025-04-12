import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = load_model("model.h5")

# Define class labels (must match your training)
class_labels = ["Apple", "Kiwi", "Orange"]

# Streamlit app layout
st.title("Fruit Image Classifier 🍎🥝🍊")
st.write("Upload an image of an apple, kiwi, or orange, and the model will predict it.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img = image.resize((128, 128))  # use the size your model was trained on
    img = img_to_array(img)
    img = img / 255.0  # normalize if your model was trained this way
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)
    predicted_class = class_labels[np.argmax(prediction)]

    # Show result
    st.success(f"Prediction: **{predicted_class}**")

