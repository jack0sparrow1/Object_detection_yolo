import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Set up page config to use a wide layout
st.set_page_config(layout="wide")

# Load YOLO model and cache it to prevent reloading on every rerun
@st.cache_resource
def load_model():
    return YOLO('my_model.pt')
model = load_model()

st.title("YOLO11 Live Webcam Object Detection")

# --- Video Processing Class ---
# This class handles the video stream from the user's browser
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        # We'll load the model here once per session
        self.model = load_model()

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert the frame to a numpy array for OpenCV/YOLO
        image = frame.to_ndarray(format="bgr24")
        
        # Perform inference on the frame
        results = self.model(image)
        
        # Draw the bounding boxes and labels on the frame
        annotated_image = results[0].plot()

        # Convert the annotated numpy array back to an av.VideoFrame
        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

# --- Streamlit UI and WebRTC Setup ---
# This single function call handles the entire webcam process
webrtc_streamer(
    key="yolo-webcam-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.info("Click 'Start' above to begin object detection. You may need to grant camera permissions.")