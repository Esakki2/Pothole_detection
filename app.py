#Model loading

from ultralytics import YOLO
model = YOLO(r"C:\Users\ESAKKI\Desktop\fp\models\content\runs\detect\train\weights\best.pt")




#web app

import io
import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Load the YOLOv8 model
model_path = (r"C:\\Users\\ESAKKI\\Desktop\\fp\\models\\content\\runs\\detect\\train\\weights\\best.pt")
model = YOLO(model_path) 
# Initialize session state for storing images
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []

# Function to perform object detection
def detect_objects(image, conf):
    results = model.predict(image, conf=conf)
    return results[0].plot()  # Return the annotated image

# Streamlit app UI
st.title("Pothole Detection using Deep Learning")

# Left Navigation Bar
page = st.sidebar.radio("Select Input:", ["Image", "File Upload", "Camera"])

# Confidence Threshold Slider
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# Input Processing
new_image = None  # Placeholder for a new image

if page == "Image":
    image_url = st.text_input("Enter Image URL")
    if image_url:
        try:
            new_image = Image.open(image_url).convert("RGB")
        except:
            st.error("Invalid image URL or unable to open image.")

elif page == "File Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        new_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

elif page == "Camera":
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        new_image = Image.open(io.BytesIO(bytes_data)).convert("RGB")

# If a new image is provided, process and store it
if new_image is not None:
    annotated_frame = detect_objects(new_image, conf_threshold)
    st.session_state.uploaded_images.append((new_image, annotated_frame))

# Display all uploaded images in a grid
if st.session_state.uploaded_images:
    for original, annotated in reversed(st.session_state.uploaded_images):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(original, caption="Original", use_container_width=True)
        with col2:
            st.subheader("Annotated Image")
            st.image(annotated, caption="Detected Objects", use_container_width=True)
