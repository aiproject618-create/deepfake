
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Deepfake Detection AI",
    page_icon="🔍",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deepfake_model.keras")

# ---------------- HEADER ----------------
st.title("🔍 Deepfake Detection AI")
st.subheader("AI-Powered Image Authenticity Checker")

st.write(
    "Upload an image to check whether the model classifies it "
    "as Real or Fake."
)

st.info(
    "This is an educational AI project. Predictions are not "
    "guaranteed to be correct."
)

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button("🔎 Detect Deepfake", type="primary"):
            with st.spinner("Analyzing image..."):
                # Prepare image for the model
                image_resized = image.resize((224, 224))
                image_array = np.array(
                    image_resized, dtype=np.float32
                )
                image_array = np.expand_dims(image_array, axis=0)

                # Model predicts probability of Real.
                # This assumes the model was trained with
                # class_names=["Fake", "Real"].
                prediction = load_model().predict(
                    image_array, verbose=0
                )[0][0]

                real_probability = float(prediction)
                fake_probability = 1.0 - real_probability

                st.divider()
                st.subheader("Detection Result")

                if real_probability >= 0.5:
                    st.success("✅ Model Prediction: REAL")
                else:
                    st.error("⚠️ Model Prediction: FAKE")

                st.write(
                    f"Real score: {real_probability * 100:.2f}%"
                )
                st.write(
                    f"Fake score: {fake_probability * 100:.2f}%"
                )

                st.progress(
                    int(round(real_probability * 100))
                )
                st.caption(
                    "The bar represents the model's Real score, "
                    "not a guarantee of authenticity."
                )

    except Exception as e:
        st.error(f"Could not process the image: {e}")

# ---------------- PROJECT INFORMATION ----------------
with st.expander("📘 About this project"):
    st.write("""
    **Project:** Deepfake Detection using Artificial Intelligence

    **Technology used:**
    - Python
    - Streamlit
    - TensorFlow and Keras
    - MobileNetV2
    - NumPy and Pillow

    **How it works:**
    1. The user uploads an image.
    2. The image is resized to 224 × 224 pixels.
    3. The trained neural network analyzes the image.
    4. The app displays the model's prediction.

    **Limitation:** Accuracy depends on the training dataset.
    This model is an educational demonstration, not a forensic
    verification tool.
    """)

st.caption("CBSE Class 12 AI Project | Deepfake Detection")
