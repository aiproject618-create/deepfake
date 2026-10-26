```python
import io

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Deepfake AI Detection",
    page_icon="🔍",
    layout="centered"
)

st.title("Deepfake AI Detection System")
st.subheader("CBSE Class 12 Artificial Intelligence Project")

st.write(
    "Upload an image to view its properties and perform "
    "basic image analysis."
)

st.warning(
    "Educational demo: This version does not use a trained "
    "deepfake detection model and cannot reliably classify "
    "images as real or fake."
)


# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG and WEBP."
)


# ---------------- IMAGE PROCESSING ----------------
if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.getvalue()

        # Verify that the uploaded file is a readable image.
        with Image.open(io.BytesIO(image_bytes)) as source:
            source.verify()

        # Reopen the image after verification.
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        st.success("Image uploaded successfully!")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        # ---------------- IMAGE DETAILS ----------------
        width, height = image.size
        file_size_kb = len(image_bytes) / 1024

        image_array = np.asarray(image, dtype=np.float32)

        brightness = float(np.mean(image_array))
        contrast = float(np.std(image_array))

        st.header("Image Information")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Width", f"{width} pixels")
            st.metric("Height", f"{height} pixels")

        with col2:
            st.metric("Total Pixels", f"{width * height:,}")
            st.metric("File Size", f"{file_size_kb:.2f} KB")

        # ---------------- BASIC IMAGE ANALYSIS ----------------
        st.header("Basic Image Analysis")

        st.write(f"Average Brightness: {brightness:.2f} / 255")

        brightness_percent = int(
            np.clip(brightness / 255 * 100, 0, 100)
        )
        st.progress(brightness_percent)

        st.write(f"Pixel Intensity Variation: {contrast:.2f}")

        st.info(
            "These measurements describe image properties only. "
            "They do not determine whether an image is real or fake."
        )

        # ---------------- DOWNLOAD REPORT ----------------
        report = (
            "DEEPFA
