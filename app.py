
import streamlit as st
from PIL import Image
from transformers import pipeline
import torch

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Deepfake AI Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Deepfake AI Detection System")
st.write(
    "A CBSE Class 12 AI project to classify uploaded "
    "images as real or potentially deepfake."
)

st.info(
    "Educational prototype only. Predictions are not "
    "proof that an image is genuine or fake."
)

# -------------------------------
# Load Pretrained Model
# -------------------------------
@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        "image-classification",
        model="dima806/deepfake_vs_real_image_detection",
        device=device
    )

# -------------------------------
# Image Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload an image to analyze",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                classifier = load_model()
                predictions = classifier(image)

            # Show the model's raw predictions
            st.subheader("Detection Results")

            for result in predictions:
                label = result["label"]
                score = float(result["score"])

                st.write(f"**{label}**")
                st.progress(max(0.0, min(1.0, score)))
                st.write(f"Model score: {score * 100:.2f}%")

            # Interpret common label names
            best = predictions[0]
            label = best["label"].lower()

            if "fake" in label or "deepfake" in label:
                st.error(
                    "The model classifies this image as "
                    "potentially fake or manipulated."
                )
            elif "real" in label or "authentic" in label:
                st.success(
                    "The model classifies this image as real."
                )
            else:
                st.warning(
                    "The model returned an unfamiliar label. "
                    "Review the raw predictions above."
                )

            st.caption(
                "Scores reflect the model's output, not a "
                "guaranteed probability of authenticity."
            )

    except Exception as error:
        st.error("Unable to analyze the image.")
        st.write(
            "Check your internet connection and installed "
            "libraries, then try again."
        )
        st.caption(f"Technical details: {error}")

# -------------------------------
# Project Information
# -------------------------------
with st.expander("About This Project"):
    st.write("""
    **Project:** Deepfake AI Detection System

    **Objective:** Demonstrate how a pretrained AI model
    classifies images as real or fake.

    **Technology used:**
    - Python
    - Streamlit
    - Hugging Face Transformers
    - PyTorch
    - Pillow

    **Limitation:** Results depend on the model's training
    data and may be incorrect for unfamiliar images.
    This prototype analyzes still images, not videos.
    """)
