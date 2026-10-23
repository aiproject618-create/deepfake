
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
from pathlib import Path

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Deepfake Detection AI",
    page_icon="🔍",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "deepfake_model.keras"
IMAGE_SIZE = (224, 224)
THRESHOLD = 0.5

# ---------------- LOAD TRAINED MODEL ----------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH.name}. "
            "Place deepfake_model.keras beside app.py."
        )
    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

# ---------------- HEADER ----------------
st.title("🔍 Deepfake Detection AI")
st.subheader("AI-Powered Image Authenticity Checker")

st.write(
    "Upload a face image to see whether the trained AI model "
    "classifies it as real or fake."
)

st.warning(
    "Educational project only. The prediction is not proof "
    "that an image is genuine or manipulated."
)

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                model = load_model()

                # Match the input size used during training.
                resized = image.resize(IMAGE_SIZE)
                pixels = np.asarray(
                    resized, dtype=np.float32
                )
                batch = np.expand_dims(pixels, axis=0)

                # IMPORTANT:
                # This assumes training used
                # class_names=["Fake", "Real"].
                # With binary labels, Fake=0 and Real=1.
                score = float(
                    model.predict(batch, verbose=0).reshape(-1)[0]
                )

                if not np.isfinite(score) or not 0 <= score <= 1:
                    st.error("The model returned an invalid score.")
                else:
                    fake_score = 1.0 - score
                    real_score = score

                    st.divider()
                    st.subheader("Detection Result")

                    if score >= THRESHOLD:
                        st.success("Prediction: REAL")
                    else:
                        st.error("Prediction: FAKE")

                    st.write("Model output scores (not verified probabilities):")
                    st.write(f"Real score: {real_score * 100:.2f}%")
                    st.write(f"Fake score: {fake_score * 100:.2f}%")

                    st.caption(
                        "The model's scores may be poorly calibrated "
                        "and can be wrong on unfamiliar images."
                    )

                    st.progress(round(real_score * 100))

    except (UnidentifiedImageError, OSError):
        st.error("The uploaded file is not a valid image.")
    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Could not analyze the image: {e}")

# ---------------- ABOUT PROJECT ----------------
with st.expander("About this project"):
    st.markdown("""
    **Project:** Deepfake Detection using Artificial Intelligence

    **Technologies:** Python, Streamlit, TensorFlow, Keras,
    MobileNetV2, NumPy and Pillow.

    **Working:**
    1. The user uploads an image.
    2. The image is resized to 224 × 224 pixels.
    3. The trained model processes the image.
    4. The app displays its prediction and scores.

    **Limitation:** Performance depends on the training dataset.
    This app is not a forensic verification system.
    """)

st.caption("CBSE Class 12 AI Project | Deepfake Detection")
