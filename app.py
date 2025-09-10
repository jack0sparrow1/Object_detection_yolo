from flask import Flask, request, render_template
import numpy as np
import cv2
import base64
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model (Ultralytics-style)
model = YOLO('my_model.pt')

def inference_img(image_bytes):
    # Convert bytes to numpy image (RGB)
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run inference
    results = model(img_rgb)

    # Render annotated image
    annotated_img = results[0].plot()  # Ultralytics API

    # Convert annotated image to base64 for HTML
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    encoded_img = base64.b64encode(buffer).decode('utf-8')
    return encoded_img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    img_data = None
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            img_bytes = file.read()
            img_data = inference_img(img_bytes)
    return render_template('index.html', img_data=img_data)

if __name__ == '__main__':
    app.run(debug=True)
