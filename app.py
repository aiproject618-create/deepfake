```python
import os
import streamlit as st
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Deepfake AI Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Deepfake AI Detection System")
st.write("CBSE Class 12 Artificial Intelligence Project")
st.write(
    "Upload an image to analyze it using a trained deep learning model."
)

MODEL_PATH = "deepfake_model.keras"


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model(model_path):
    import tensorflow as tf
    return tf.keras.models.load_model(
        model_path,
        compile=False
    )


model = None

if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
    except Exception:
        st.error(
            "The model file could not be loaded. "
            "Check that deepfake_model.keras is a valid Keras model."
        )
else:
    st.warning(
        "Trained model not found. Add deepfake_model.keras "
        "to your project folder to enable predictions."
    )


# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)


# ---------------- IMAGE PREPROCESSING ----------------
def prepare_image(image, model):
    input_shape = model.input_shape

    if isinstance(input_shape, list) or len(input_shape) != 4:
        raise ValueError("The model must accept one image input.")

    _, height, width, channels = input_shape

    if height is None or width is None:
        raise ValueError("The model must have a fixed input size.")

    if channels not in (1, 3):
        raise ValueError("Only grayscale and RGB models are supported.")

    image = ImageOps.exif_transpose(image)

    if channels == 1:
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    image = image.resize(
        (int(width), int(height)),
        Image.Resampling.BILINEAR
    )

    array = np.asarray(image, dtype=np.float32)

    if channels == 1:
        array = np.expand_dims(array, axis=-1)

   
