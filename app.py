import streamlit as st
from ultralytics import YOLO
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import logging

st.set_page_config(layout="wide")
st.title("YOLO11 Live Webcam Object Detection")

@st.cache_resource
def load_model():
    return YOLO('my_model.pt')  # Replace with your model path

model = load_model()

class YOLOVideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            results = self.model(img)
            img_annotated = results[0].plot()
            return av.VideoFrame.from_ndarray(img_annotated, format="bgr24")
        except Exception as e:
            logging.error(f"YOLO inference error: {e}")
            # Return original frame if error occurs
            return frame

webrtc_streamer(
    key="yolo-live-stream",
    video_processor_factory=YOLOVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    async_processing=True,
)

st.info("Click 'Start' to initiate webcam. Allow camera permission when prompted.")
