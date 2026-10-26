
import os
import streamlit as st
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Deepfake AI Detector",
    page_icon="🔍",
    layout="centered"
)

MODEL_PATH = "deepfake_model.keras"

st.title("🔍 Deepfake AI Detection System")
st.write(
    "A Class 12 Artificial Intelligence project to analyze "
    "images using a trained deep learning model."
)

st.info(
    "Upload an image to analyze it. The prediction is generated "
    "by a trained model, not by a random guess."
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_detection_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)


model = None
model_error = None

if os.path.isfile(MODEL_PATH):
    try:
       
