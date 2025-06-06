import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import os
from model.utils import load_model, predict_disease  # assume you moved helpers here

# --- Load model ---
@st.cache_resource
def load_trained_model():
    model_path = 'model/plant_cnn.pt'
    return load_model(model_path)

model = load_trained_model()

st.write("Model loaded successfully.")
st.write("Model architecture:", model.__class__.__name__)

# Streamlit app for Plant Disease Detection
# --- Title ---
st.title("🌿 Plant Disease Detector")
st.subheader("Upload a leaf image and let AI do the diagnosis.")

# --- Upload ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # --- Preprocess ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)

    # --- Predict ---
    with st.spinner("Analyzing..."):
        prediction, confidence = predict_disease(model, image_tensor)

    st.success(f"🧠 Predicted: **{prediction}** with {confidence:.2f}% confidence")

    if prediction == "Unknown or Uncertain":
        st.warning("⚠️ This image may not belong to any known disease class.")

