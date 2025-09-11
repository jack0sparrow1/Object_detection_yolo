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

# This class processes each video frame from webcam and applies YOLO detection
class YOLODetector(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            # Convert video frame to numpy array compatible with OpenCV (BGR)
            img = frame.to_ndarray(format="bgr24")

            # Run YOLO inference, specifying a smaller image size for speed
            results = self.model(img, imgsz=320) # Optimization

            # Draw bounding boxes and labels on the frame
            annotated_img = results[0].plot()

            # Convert back to video frame for streaming output
            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
        except Exception as e:
            # Log errors without stopping video stream
            logging.error(f"Error during frame processing: {e}")
            return frame  # Return original frame if error

# Streamlit-webrtc call: starts the webcam video stream with YOLO Detector
webrtc_streamer(
    key="yolo-live-webcam",
    video_processor_factory=YOLODetector,
    media_stream_constraints={"video": True, "audio": False},  # webcam only, no mic
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},  # STUN server
            {"urls": ["turn:openrelay.metered.ca:80"], "username": "openrelayproject", "credential": "openrelayproject"} # Free TURN server
        ]
    },
    async_processing=True,  # process frames asynchronously for smoother video
)

st.info("Click 'Start' button above to begin live object detection using your webcam. "
        "Please allow camera permissions in your browser.")