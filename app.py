import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model("model.h5", compile=False)


st.title("Fruit Classifier (with Numeric Inputs)")

st.write("Enter the 3 features below:")

# Custom feature names
feature_names = ["Size", "Color", "Shape"]

# Display predefined numeric values for each fruit
st.write("Example values for each fruit:")

# Define some example values for each fruit
fruit_examples = {
    "Kiwi": {"Size": 0.04, "Color": 0.15, "Shape": 0.18},
    "Apple": {"Size": 0.08, "Color": 0.1, "Shape": 0.1},
    "Orange": {"Size": 0.09, "Color": 0.25, "Shape": 0.12}
}

# Show the example values in a table
st.write("Example Sizes, Colors, and Shapes for each fruit:")

# Create a table displaying these example values
fruit_df = {
    "Fruit": list(fruit_examples.keys()),
    "Size": [f"{fruit_examples[fruit]['Size']}" for fruit in fruit_examples],
    "Color": [f"{fruit_examples[fruit]['Color']}" for fruit in fruit_examples],
    "Shape": [f"{fruit_examples[fruit]['Shape']}" for fruit in fruit_examples],
}

# Display the table using Streamlit's markdown
st.write(fruit_df)

# Create inputs for Size, Color, and Shape
features = []
for i in range(3):
    value = st.number_input(feature_names[i], value=0.0)
    features.append(value)

# Convert to NumPy array and reshape
features_array = np.array([features])  # shape (1, 3)

if st.button("Predict"):
    # Make the prediction
    prediction = model.predict(features_array)
    
    # Get the predicted class (which fruit has the highest probability)
    predicted_class_index = np.argmax(prediction)  # index of the highest probability
    
    # Map the index to fruit names
    fruit_classes = ["Kiwi", "Apple", "Orange"]
    predicted_class = fruit_classes[predicted_class_index]
    
    st.write("Prediction:", predicted_class)
