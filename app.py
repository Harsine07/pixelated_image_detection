import streamlit as st
import os
import numpy as np
import tensorflow as tf
import warnings  # Import warnings module
from keras.preprocessing.image import img_to_array
from PIL import Image

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load the trained model
MODEL_PATH = "pixelation_detection_model.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Image processing parameters
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Streamlit UI
st.title("Pixelated Image Detector")
st.write("Upload an image to check if it is pixelated.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

def predict_pixelation(image):
    """Predict if an image is pixelated or not"""
    img = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = img_to_array(img) / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)[0][0]
    return "Not Pixelated" if prediction < 0.2 else "Pixelated"


# Display uploaded image and prediction
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)  # Updated parameter
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if st.button("Predict"):
        result = predict_pixelation(image)
        st.write(f"<h3 style='text-align: center;'>{result}</h3>", unsafe_allow_html=True)


    
