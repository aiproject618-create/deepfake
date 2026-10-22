
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(
    page_title="Deepfake AI Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Deepfake AI Detection System")
st.write("CBSE Class 12 Artificial Intelligence Project")

st.info(
    "This app uses a trained AI model. Results may be incorrect "
    "and should not be treated as proof of authenticity."
)

MODEL_PATH = Path(__file__).parent / "deepfake_model.keras"

# -------------------------------
# Load trained model
# -------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "deepfake_model.keras was not found. "
            "Train your model and add the file to your project."
        )

    return tf.keras.models.load_model(MODEL_PATH)

# -------------------------------
# Image upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
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

                # Match the training image size and pixel range.
                resized = image.resize((224, 224))
                image_array = np.asarray(
                    resized, dtype=np.float32
                )
                image_array = np.expand_dims(
                    image_array, axis=0
                )

                # The training model outputs sigmoid probability
                # for class 1: real.
                prediction = model.predict(
                    image_array, verbose=0
                )

                if prediction.size != 1:
                    st.error(
                        "Unexpected model output. "
                        "This app requires a binary classifier "
                        "with one sigmoid output."
                    )
                    st.stop()

                real_score = float(prediction.reshape(-1)[0])

                if not np.isfinite(real_score) or not (
                    0.0 <= real_score <= 1.0
                ):
                    st.error("The model returned an invalid score.")
                    st.stop()

                fake_score = 1.0 - real_score

            st.subheader("Analysis Results")

            col1, col2 = st.columns(2)
            col1.metric("Real score", f"{real_score:.1%}")
            col2.metric("Fake score", f"{fake_score:.1%}")

            st.write("Model score comparison")
            st.progress(real_score)

            # Use a cautious uncertainty band.
            if 0.40 <= real_score <= 0.60:
                st.warning(
                    "Inconclusive result: the model scores are "
                    "too close to make a confident classification."
                )
            elif real_score > 0.60:
                st.success("Prediction: REAL")
            else:
                st.error("Prediction: POTENTIALLY FAKE")

            st.caption(
                "These scores are model outputs, not necessarily "
                "calibrated probabilities. The uncertainty band "
                "does not guarantee detection accuracy."
            )

    except FileNotFoundError as error:
        st.error(str(error))

    except Exception as error:
        st.error("Could not analyze this image.")
        st.code(str(error))

# -------------------------------
# About section
# -------------------------------
with st.expander("About this project"):
    st.write("""
    This project demonstrates binary image classification
    using a TensorFlow model trained with transfer learning.

    Limitations:
    - Performance depends on training data quality.
    - Some real images may be classified as fake.
    - Some manipulated images may be classified as real.
    - This prototype does not analyze videos.
    """)
