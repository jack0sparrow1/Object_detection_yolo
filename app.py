import streamlit as st
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import logging

# Set wide layout for better video display
st.set_page_config(layout="wide")
st.title("YOLO11 Live Webcam Object Detection")

# Cache model loading so it happens once per session
@st.cache_resource
def load_yolo_model():
    return YOLO('my_model.pt')  # Replace with your actual trained weights path

model = load_yolo_model()
# Define a custom video transformer for object detection
class ObjectDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        # Load your object detection model here (e.g., YOLO, OpenCV DNN, etc.)
        pass

    def transform(self, frame):
        # Convert the frame to a numpy array
        img = frame.to_ndarray(format="bgr24")

        # Perform object detection on the frame (replace this with your model's logic)
        # Example: Draw a rectangle on the frame
        height, width, _ = img.shape
        cv2.rectangle(img, (50, 50), (width - 50, height - 50), (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit app UI
st.title("Live Object Detection with Streamlit")
st.info("Click 'Start' below to begin live object detection using your webcam. "
        "Please allow camera permissions in your browser.")

# WebRTC streamer for live video feed
webrtc_streamer(
    key="object-detection",
    video_transformer_factory=ObjectDetectionTransformer,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},  # STUN server
            {"urls": ["turn:openrelay.metered.ca:80"], "username": "openrelayproject", "credential": "openrelayproject"}  # Free TURN server
        ]
    },
    async_processing=True,  # Process frames asynchronously for smoother video
)

st.info("Click 'Start' button above to begin live object detection using your webcam. "
        "Please allow camera permissions in your browser.")