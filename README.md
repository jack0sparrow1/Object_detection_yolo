# YOLO Object Detection Project with Streamlit Deployment

## Overview
This project implements a custom object detection pipeline using the YOLO11 model. It covers:
- Automated image collection from webcam with Python
- Labeling images using Label Studio
- Dataset preparation with train/validation splits
- Training YOLO11 on Google Colab with GPU acceleration
- Local inference with live webcam predictions
- Streamlit-based live webcam detection web app deployment

## Features
- Automated data collection and labeling
- Training on custom datasets using YOLO11
- Real-time webcam detection locally and remotely via Streamlit
- Model export and reuse

## Project Structure
data/images/ # Collected raw images
data/labels/ # YOLO format label files
train/ # Trained model from google colab
yolo_detect.py # Local webcam detection script
streamlit_app.py # Streamlit live webcam detection app
requirements.txt # Required Python packages
README.md # Project overview

text

## Setup and Usage

### 1. Image Collection and Labeling
- Run the Python script to capture images from your webcam.
- Label images with Label Studio and export as zip file with annotations.

### 2. Prepare Dataset
- Extract exported dataset.
- Use provided script to split into training and validation folders.
- Create `data.yaml` file defining dataset paths and classes.

### 3. Model Training (Google Colab)
- Upload dataset and `data.yaml` to Google Drive.
- Train YOLO11 model with GPU support on Colab.
- Validate on held-out data.
- This is my colab link: [Google Colab link to train](https://colab.research.google.com/drive/17tz7Qf3Dcow2mSPwtJVD4yCz9NzBm8kx?usp=sharing)

### 4. Local Inference
- Download and unzip trained weights/model.
- Run `yolo_detect.py` to perform real-time webcam detection on your PC.

### 5. Streamlit Deployment
- Run `streamlit run streamlit_app.py` to start live detection app.
- Deploy on Render or similar for free remote access.

## Dependencies
- Python 3.x
- OpenCV
- Ultralytics YOLO11
- Streamlit
- PyYAML

Install dependencies via:

pip install -r requirements.txt

text

## Notes
- Streamlit app handles webcam video capture in the browser for live remote inference.
- Performance depends on hardware and network conditions.
- Free Render plans support lightweight web apps suitable for demos.

## License
- YOLO11 model is under AGPL-3.0 License.
- Other dependencies have their respective open-source licenses.

---

Feel free to open issues or contribute improvements.