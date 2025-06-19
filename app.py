import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load model
model = tf.keras.models.load_model('models/resampled_model.h5')

# Set page config
st.set_page_config(page_title="DB-VAE Face Classifier", layout="centered")

st.title("🧠 DB-VAE Fairness-Aware Face Classifier")
st.write("Upload an image to classify it as a **Face** or **Not Face** using a fairness-aware CNN trained with adaptive sampling.")

uploaded_file = st.file_uploader("Upload an image (64x64 grayscale)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Read image
    image = Image.open(uploaded_file).convert('L')  # convert to grayscale
    image = image.resize((64, 64))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 64, 64, 1)

    st.image(image, caption="Uploaded Image", use_container_width="auto")

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "Face" if prediction > 0.5 else "Not Face"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.markdown(f"### Prediction: `{label}`")
    st.markdown(f"**Confidence:** `{confidence:.2%}`")
