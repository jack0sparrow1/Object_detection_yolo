import streamlit as st
import cv2
from ultralytics import YOLO

# Set up page config to use a wide layout
st.set_page_config(layout="wide")

# Load YOLO model and cache it to prevent reloading on every rerun
@st.cache_resource
def load_model():
    return YOLO('my_model.pt')
model = load_model()

st.title("YOLO11 Live Webcam Object Detection")

# Initialize session state for the webcam to control its state
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False

# Create a single checkbox and link it to the session state
run = st.checkbox('Run Webcam', value=st.session_state.webcam_running)

# Update the session state when the checkbox is clicked
st.session_state.webcam_running = run

FRAME_WINDOW = st.image([])
cap = None

if st.session_state.webcam_running:
    if cap is None:
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.warning("Failed to open webcam.")
    else:
        while st.session_state.webcam_running:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture webcam frame")
                break
            
            # Convert BGR frame to RGB
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Inference and visualization
            results = model(img)
            annotated_frame = results[0].plot()
            
            FRAME_WINDOW.image(annotated_frame)
            
        cap.release()
        st.write('Webcam stopped.')
        
else:
    st.write('Webcam stopped.')
    if cap and cap.isOpened():
        cap.release()