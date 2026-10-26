
import streamlit as st
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Deepfake AI Detection",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Deepfake AI Detection System")
st.subheader("CBSE Class 12 Artificial Intelligence Project")

st.write(
    "Upload an image to view its properties and perform "
    "a basic image-quality analysis."
)

st.warning(
    "Demo mode: This version does not use a trained deepfake "
    "detection model. It cannot reliably classify an image as real or fake."
)

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, WEBP"
)

# ---------------- IMAGE ANALYSIS ----------------
if uploaded_file is not None:
    try:
        with Image.open(uploaded_file) as original:
            original.verify()

        uploaded_file.seek(0)

        with Image.open(uploaded_file) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")

        st.success("Image uploaded successfully.")

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

        # Image properties
        width, height = image.size
        image_array = np.asarray(image)

        st.subheader("Image Information")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Width", f"{width} pixels")
            st.metric("Height", f"{height} pixels")

        with col2:
            st.metric("Total pixels", f"{width * height:,}")
            st.metric("File size", f"{uploaded_file.size / 1024:.2f} KB")

        # Basic image statistics
        st.subheader("Basic Image Analysis")

        brightness = float(np.mean(image_array))
        contrast = float(np.std(image_array))

        st.write(f"Average pixel brightness: {brightness:.2f} / 255")
        st.progress(int(np.clip(brightness / 255 * 100, 0, 100)))

        st.write(f"Pixel intensity variation: {contrast:.2f}")

        st.info(
    "These measurements describe the image only. They do not "
    "indicate whether the image is genuine or AI-generated."
)
