```python
import io
import streamlit as st
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# Page configuration
st.set_page_config(
    page_title="Deepfake AI Detection",
    page_icon="🔍",
    layout="centered"
)

# App title
st.title("Deepfake AI Detection System")
st.subheader("CBSE Class 12 AI Project")

st.write("Upload an image to view its basic properties.")

st.warning(
    "Demo mode: No trained deepfake detection model is connected. "
    "This app cannot reliably classify images as real or fake."
)

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

# Process image
if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.getvalue()

        # Validate image
        with Image.open(io.BytesIO(image_bytes)) as source:
            source.verify()

        # Read image
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        st.success("Image uploaded successfully!")

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

        # Calculate image properties
        width, height = image.size
        file_size = len(image_bytes) / 1024
        pixels = np.asarray(image, dtype=np.float32)

        brightness = float(np.mean(pixels))
        contrast = float(np.std(pixels))

        # Display results
        st.header("Image Information")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Width", f"{width} pixels")
            st.metric("Height", f"{height} pixels")

        with col2:
            st.metric("Total pixels", f"{width * height:,}")
            st.metric("File size", f"{file_size:.2f} KB")

        st.header("Basic Image Analysis")

        st.write(f"Average brightness: {brightness:.2f} / 255")
        st.progress(int(np.clip(brightness / 255 * 100, 0, 100)))

        st.write(f"Pixel intensity variation: {contrast:.2f}")

        st.info(
            "These measurements describe image properties only. "
            "They do not detect deepfakes."
        )

        st.download_button(
            label="Download Image Report",
            data=(
                f"Image: {uploaded_file.name}\n"
                f"Dimensions: {width} x {height}\n"
                f"Total pixels: {width * height}\n"
                f"File size: {len(image_bytes)} bytes\n"
                f"Average brightness: {brightness:.2f}\n"
                f"Pixel intensity variation: {contrast:.2f}\n"
                "Deepfake classification: Not performed\n"
            ),
            file_name="image_report.txt",
            mime="text/plain"
        )

    except (UnidentifiedImageError, OSError, ValueError):
        st.error(
            "Invalid image. Please upload a valid JPG, JPEG, PNG, "
            "or WEBP file."
        )

    except Exception as error:
        st.error(f"An unexpected error occurred: {error}")

# Footer
st.divider()
st.caption("CBSE Class 12 | Artificial Intelligence Project")
```
