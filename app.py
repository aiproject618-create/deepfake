
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pathlib import Path

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Deepfake Detection AI",
    page_icon="🔍",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "deepfake_model.keras"
IMAGE_SIZE = (224, 224)
THRESHOLD = 0.5

# ==================================================
# LOAD THE TRAINED MODEL
# ==================================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "deepfake_model.keras was not found. "
            "Place it in the same folder as app.py."
        )

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

# ==================================================
# HEADER
# ==================================================
st.title("🔍 Deepfake Detection AI")
st.subheader("AI-Powered Image Authenticity Checker")

st.markdown(
    """
    Upload an image to see how the trained AI model
    classifies it.
    """
)

st.info(
    "This is an educational AI project. Its predictions "
    "may be incorrect and should not be treated as proof "
    "that an image is genuine or manipulated."
)

# ==================================================
# IMAGE UPLOAD
# ==================================================
uploaded_file = st.file_uploader(
    "Choose an image from your device",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False
)

if uploaded_file is not None:

    try:
        # Read uploaded bytes and validate the image
        image_bytes = uploaded_file.getvalue()

        if not image_bytes:
            st.error("The uploaded file is empty.")

        else:
            import io

            with Image.open(io.BytesIO(image_bytes)) as check:
                check.verify()

            with Image.open(io.BytesIO(image_bytes)) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

            st.caption(
                f"Image size: {image.width} × {image.height} pixels"
            )

            # ==========================================
            # PREDICTION
            # ==========================================
            if st.button("🔎 Detect Deepfake", type="primary"):

                try:
                    with st.spinner("Analyzing image..."):

                        model = load_model()

                        # Resize to the model's expected input size
                        resized_image = image.resize(IMAGE_SIZE)

                        image_array = np.asarray(
                            resized_image,
                            dtype=np.float32
                        )

                        # Add batch dimension
                        image_batch = np.expand_dims(
                            image_array, axis=0
                        )

                        # Model output for binary classification
                        prediction = model.predict(
                            image_batch,
                            verbose=0
                        )

                        score = float(
                            np.asarray(prediction).reshape(-1)[0]
                        )

                        if not np.isfinite(score) or not 0 <= score <= 1:
                            st.error(
                                "The model returned an invalid score. "
                                "Check that it is a binary classifier "
                                "with a sigmoid output."
                            )

                        else:
                            # Assumes Fake=0 and Real=1 in training
                            real_score = score
                            fake_score = 1.0 - score

                            st.divider()
                            st.subheader("📊 Detection Result")

                            if real_score >= THRESHOLD:
                                st.success("Prediction: REAL")
                            else:
                                st.error("Prediction: FAKE")

                            col1, col2 = st.columns(2)

                            with col1:
                                st.metric(
                                    "Real score",
                                    f"{real_score * 100:.2f}%"
                                )

                            with col2:
                                st.metric(
                                    "Fake score",
                                    f"{fake_score * 100:.2f}%"
                                )

                            st.write("Real score")
                            st.progress(
                                int(round(real_score * 100))
                            )

                            st.caption(
                                "These are model scores, not guaranteed "
                                "probabilities or proof of authenticity."
                            )

                except FileNotFoundError as e:
                    st.error(str(e))

                except Exception as e:
                    st.error(
                        f"Prediction failed: {type(e).__name__}: {e}"
                    )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ) as e:
        st.error(
            "This file could not be decoded as an image. "
            "Please try a normal JPG or PNG image. "
            f"Details: {type(e).__name__}: {e}"
        )

    except Exception as e:
        st.error(
            f"Image processing failed: {type(e).__name__}: {e}"
        )

# ==================================================
# ABOUT THE PROJECT
# ==================================================
with st.expander("📘 About this project"):

    st.markdown(
        """
        **Project:** Deepfake Detection using Artificial Intelligence

        **Technologies used:**
        - Python
        - Streamlit
        - TensorFlow / Keras
        - MobileNetV2
        - NumPy and Pillow

        **How it works:**
        1. The user uploads an image.
        2. The app validates and resizes the image.
        3. The trained neural network analyzes the image.
        4. The app displays the model's classification and scores.

        **Limitations:** Accuracy depends on the training data,
        model architecture, and preprocessing. A model can make
        mistakes on images unlike those used during training.
        """
    )

st.divider()
st.caption("CBSE Class 12 AI Project | Deepfake Detection")
