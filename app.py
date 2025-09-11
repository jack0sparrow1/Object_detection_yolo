import streamlit as st
from ultralytics import YOLO
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import logging

st.set_page_config(layout="wide")
st.title("YOLO11 Live Webcam Object Detection")

# Cache the model to prevent reloading on every rerun
@st.cache_resource
def load_model():
    return YOLO('my_model.pt')

model = load_model()

# Video processing class to apply YOLO detection on webcam frames
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            image = frame.to_ndarray(format="bgr24")
            results = self.model(image)
            annotated_image = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")
        except Exception as e:
            logging.error(f"Error during inference: {e}")
            return frame  # Return original frame on error

webrtc_streamer(
    key="yolo-live-webcam",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    async_processing=True,
)

st.info(
    "Click 'Start' above to begin live object detection.\n"
    "Grant camera permissions when prompted."
)
