```python
import streamlit as st
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import io

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
    "basic image analysis."
)

st.warning(
    "Demo mode: This application does not use a trained deepfake "
    "detection model. It cannot reliably classify images as real or fake."
)

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)

# ---------------- PROCESS IMAGE ----------------
if uploaded_file is not None:

    try:
        image_bytes = uploaded_file.getvalue()

        with Image.open(io.BytesIO(image_bytes)) as source:
            source.verify()

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
            st.metric("Width", f"{width} px")
            st.metric("Height", f"{height} px")

        with col2:
            st.metric("Total Pixels", f"{width * height:,}")
            st.metric("File Size", f"{file_size_kb:.2f} KB")

        st.header("Basic Image Analysis")

        st.write(f"Average Brightness: {brightness:.2f} / 255")
        st.progress(int(np.clip(brightness / 255 * 100, 0, 100)))

        st.write(f"Pixel Intensity Variation: {contrast:.2f}")

        st.info(
            "These measurements describe the image only. "
            "They do not determine whether the image is real or fake."
        )

        # ---------------- DOWNLOAD REPORT ----------------
        report = (
            "DEEPFAKE AI PROJECT - IMAGE ANALYSIS REPORT\n\n"
            f"Filename: {uploaded_file.name}\n"
            f"Width: {width} pixels\n"
            f"Height: {height} pixels\n"
            f"Total pixels: {width * height}\n"
            f"File size: {len(image_bytes)} bytes\n"
            f"Average brightness: {brightness:.2f}\n"
            f"Pixel intensity variation: {contrast:.2f}\n\n"
            "Deepfake detection: NOT PERFORMED\n"
            "Reason: No trained detection model is connected.\n"
        )

        st.download_button(
            label="Download Analysis Report",
            data=report,
            file_name="image_analysis_report.txt",
            mime="text/plain"
        )

    except (UnidentifiedImageError, OSError, ValueError):
        st.error(
            "The uploaded file could not be read as a valid image. "
            "Please upload a valid JPG, JPEG, PNG, or WEBP
