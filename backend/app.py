from flask import Flask, request, send_file
from flask_cors import CORS
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import io
import os
import uuid

app = Flask(__name__)

# ✅ Allow specific frontend origin
CORS(app, origins=["https://sem6-mini-project.vercel.app"])

# ✅ Manually ensure CORS headers after every response
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'https://sem6-mini-project.vercel.app'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# ✅ Check for model
MODEL_DIR = os.path.join("container", "weights")
MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Make sure it's downloaded before starting the app.")

# ✅ Load YOLO model
model = YOLO(MODEL_PATH)

@app.route('/')
def home():
    return "✅ Flask YOLO backend is running!"

# ✅ Allow POST + OPTIONS for CORS preflight
@app.route('/detect', methods=['POST', 'OPTIONS'])
def detect():
    if request.method == 'OPTIONS':
        return '', 204

    if 'image' not in request.files:
        return 'No image provided', 400

    file = request.files['image']
    image_bytes = file.read()
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    os.makedirs("temp", exist_ok=True)
    temp_filename = f"temp/{uuid.uuid4().hex}.jpg"
    cv2.imwrite(temp_filename, img)

    results = model(temp_filename, device='cpu')
    output_img = results[0].plot()

    pil_img = Image.fromarray(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG')
    buf.seek(0)

    os.remove(temp_filename)
    return send_file(buf, mimetype='image/jpeg')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # for Render
    app.run(debug=False, host='0.0.0.0', port=port)
