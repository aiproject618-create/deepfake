# Deepfake AI Detection System

**CBSE Class 12 AI / Computer Science Project**

A beginner-friendly Streamlit web application that uses a pretrained
image-classification model to classify an uploaded image as potentially
real or fake/manipulated.

> **Important:** This is an educational prototype, not a forensic tool.
> Predictions can be wrong and should not be treated as proof that an
> image is real or fake.

## Features

-   Upload an image in JPG, JPEG, PNG, or WEBP format.
-   Preview the uploaded image.
-   Run a pretrained image-classification model.
-   Display the model's labels and scores.
-   Show a simple interpretation of the top prediction.
-   Explain the project's purpose and limitations.

## Project Structure

``` text
Deepfake-Detection/
├── app.py
├── requirements.txt
└── README.md
```

## Technologies Used

-   **Python** --- programming language
-   **Streamlit** --- web application interface
-   **Hugging Face Transformers** --- loads the pretrained
    image-classification pipeline
-   **PyTorch** --- model inference backend
-   **Pillow (PIL)** --- opens and processes uploaded images

## Requirements

-   Python 3.9 or a compatible Python version supported by the installed
    libraries
-   Internet access for the initial model download
-   Enough disk space and memory to install and run PyTorch and the
    model

## Installation and Setup

### 1. Open the project folder

Open Command Prompt or a terminal in the folder containing `app.py` and
`requirements.txt`.

### 2. (Recommended) Create a virtual environment

**Windows:**

``` bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

``` bash
streamlit run app.py
```

Streamlit should open the application in your web browser. If it does
not, open the local URL printed in the terminal (usually
`http://localhost:8501`).

## How to Use

1.  Start the app using `streamlit run app.py`.
2.  Upload an image using the file uploader.
3.  Check the image preview.
4.  Click **Analyze Image**.
5.  Review the model's labels and scores.
6.  Interpret the result cautiously; the model can make mistakes.

The first run may take longer because the pretrained model must be
downloaded. An internet connection may be needed for that download.

## How It Works

1.  The user uploads an image.
2.  Pillow converts the image to RGB format.
3.  The Transformers image-classification pipeline sends the image to
    the pretrained model.
4.  The model returns one or more labels with scores.
5.  The app displays the predictions in the Streamlit interface.

The code uses the model identifier
`dima806/deepfake_vs_real_image_detection` from Hugging Face. Check the
model's documentation and label names before relying on its output.

## Limitations and Responsible Use

-   A model score is **not necessarily a calibrated probability** that
    an image is fake or real.
-   Images may be misclassified, especially if they differ from the
    model's training data.
-   The app analyzes still images; it does not analyze an entire video.
-   Do not use this result as the sole basis for accusations,
    disciplinary action, or other high-impact decisions.
-   Avoid uploading private or sensitive images unless you understand
    how and where the application is running and how the data is
    handled.

## Possible Future Improvements

-   Add video frame extraction and frame-by-frame analysis.
-   Compare results from more than one model.
-   Add a history page for test results.
-   Improve the interface with confidence explanations and sample
    images.
-   Evaluate the model on a labelled test dataset and report accuracy,
    precision, recall, and a confusion matrix.

## CBSE Project Viva: Quick Questions

**1. What is a deepfake?**\
A deepfake is synthetic or manipulated media created or altered using AI
techniques.

**2. What is Streamlit?**\
Streamlit is a Python framework used to build interactive web
applications.

**3. What is a pretrained model?**\
A pretrained model is a model that has already learned patterns from a
dataset before being used in an application.

**4. Does the app always detect fake images correctly?**\
No. The result depends on the model and image quality, and both false
positives and false negatives are possible.

**5. What is the main limitation of this project?**\
It is a basic image-classification prototype and cannot guarantee
authenticity or reliably detect every type of manipulation.

## Author

Student Name: *Add your name here*\
Class: XII\
Subject: *Add your subject here*\
School: *Add your school name here*
