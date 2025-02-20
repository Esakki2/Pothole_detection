import streamlit as st
import piexif
import json
import requests  
import folium
from streamlit_folium import st_folium
from ultralytics import YOLO
import io
import cv2
from PIL import Image
import numpy as np

# ✅ Load the YOLO model
model_path = r"C:\\Users\\ESAKKI\\Desktop\\fp\\models\\content\\runs\\detect\\train\\weights\\best.pt"
model = YOLO(model_path) 

# ✅ API Details for Location
TOKEN = "6d950b562816a0"  # Your IPinfo token
IP_ADDRESS = "1.38.96.65"  # Change this to get location for a specific IP

# ✅ Function to get geolocation (only for camera option)
def get_location():
    try:
        url = f"https://ipinfo.io/{IP_ADDRESS}?token={TOKEN}"  
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            loc = data.get("loc", "0,0").split(",")  
            return float(loc[0]), float(loc[1])
        else:
            return None, None
    except:
        return None, None

# ✅ Initialize session state for storing images
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []

# ✅ Function to perform object detection
def detect_objects(image, conf):
    results = model.predict(image, conf=conf)
    return results[0].plot()

# ✅ Streamlit UI
st.title("Pothole Detection with Deep Learning")

# ✅ Sidebar for Input Options
page = st.sidebar.radio("Select Input:", ["Image", "File Upload", "Camera"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# ✅ Input Processing (Camera + File Upload)
new_image = None  
lat, lon = None, None  # Default values (no geotagging)

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
    # ❌ No geotagging for file uploads

elif page == "Camera":
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        new_image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        # ✅ Get geolocation only for the camera
        lat, lon = get_location()

# ✅ If a new image is provided, process and store it
if new_image is not None:
    annotated_frame = detect_objects(new_image, conf_threshold)

    if lat and lon:
        # Convert GPS to EXIF format (only for Camera)
        def convert_to_exif_gps(val):
            d = int(val)
            m = int((val - d) * 60)
            s = round(((val - d) * 60 - m) * 60 * 100, 2)
            return ((d, 1), (m, 1), (int(s), 100))

        gps_ifd = {
            piexif.GPSIFD.GPSLatitude: convert_to_exif_gps(abs(lat)),
            piexif.GPSIFD.GPSLatitudeRef: 'N' if lat >= 0 else 'S',
            piexif.GPSIFD.GPSLongitude: convert_to_exif_gps(abs(lon)),
            piexif.GPSIFD.GPSLongitudeRef: 'E' if lon >= 0 else 'W'
        }

        exif_dict = {"GPS": gps_ifd}
        exif_bytes = piexif.dump(exif_dict)

        img_bytes = io.BytesIO()
        new_image.save(img_bytes, format="jpeg", exif=exif_bytes)
        img_bytes.seek(0)

        # ✅ Store image & geolocation data (only for Camera)
        st.session_state.uploaded_images.append((new_image, annotated_frame, lat, lon))
    else:
        # Store without geotagging for file upload
        st.session_state.uploaded_images.append((new_image, annotated_frame, None, None))

# ✅ Display uploaded images and geotag (Only for Camera)
if st.session_state.uploaded_images:
    for original, annotated, lat, lon in reversed(st.session_state.uploaded_images):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(original, caption="Original", use_container_width=True)
        with col2:
            st.subheader("Annotated Image")
            if lat and lon:
                st.image(annotated, caption=f"Detected at {lat}, {lon}", use_container_width=True)
            else:
                st.image(annotated, caption="Detected (No Geotag)", use_container_width=True)

    # ✅ Allow downloading only geotagged images
    if lat and lon:
        st.download_button("Download Geotagged Image", img_bytes, "geotagged_image.jpg", "image/jpeg")

st.write("📌 **Note:** Geotagging works **only** for the 'Camera' option. If location doesn't appear, ensure GPS is enabled on your device.")
