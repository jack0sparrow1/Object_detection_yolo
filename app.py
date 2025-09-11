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
    return YOLO('my_model.pt')

model = load_yolo_model()

# Define a custom video processor for object detection
# We're now using VideoTransformerBase, which is the correct class
class ObjectDetectionProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            # Convert the frame to a numpy array (BGR format)
            img = frame.to_ndarray(format="bgr24")

            # Perform object detection using YOLO, explicitly setting the image size
            results = self.model(img, imgsz=320)

            # Annotate the frame with bounding boxes
            annotated_frame = results[0].plot()

            # Convert back to a video frame and return it
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
        except Exception as e:
            # Log errors for debugging without crashing the stream
            logging.error(f"Error during frame processing: {e}")
            return frame # Return original frame on error

# Streamlit-webrtc call: starts the webcam video stream with YOLO Detector
webrtc_streamer(
    key="yolo-live-webcam",
    video_processor_factory=ObjectDetectionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["turn:openrelay.metered.ca:80"], "username": "openrelayproject", "credential": "openrelayproject"}
        ]
    },
    async_processing=True,
)

st.info("Click 'Start' above to begin live object detection using your webcam. "
        "Please allow camera permissions in your browser.")