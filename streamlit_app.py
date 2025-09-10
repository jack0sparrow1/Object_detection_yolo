import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLO model (change the path to your model weights)
model = YOLO('my_model.pt')

st.title("YOLO11 Live Webcam Object Detection")

run = st.checkbox('Run Webcam')

FRAME_WINDOW = st.image([])

# Use OpenCV to capture webcam video stream
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture webcam frame")
        break

    # Convert frame (BGR to RGB)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Inference
    results = model(img)

    # Render results on frame
    annotated_frame = results[0].plot()  # correct Ultralytics API call

    # Display in streamlit
    FRAME_WINDOW.image(annotated_frame)


if not run:
    st.write('Webcam stopped.')
    cap.release()
