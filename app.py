from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import os

app = Flask(__name__)
model = YOLO('my_model.pt')  # path to your YOLO model

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert frame to RGB for YOLO
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        annotated_frame = results[0].plot()  # draw detections

        # Convert back to BGR for OpenCV display/streaming
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', annotated_frame_bgr)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Returns a multipart response with streaming frames
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)