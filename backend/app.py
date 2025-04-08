import os
from flask import Flask, request, send_file
from flask_cors import CORS
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import io
import uuid

# Construct dynamic model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "container", "weights", "best.pt")


# Initialize Flask app and model
app = Flask(__name__)
CORS(app, origins=["https://sem6-mini-project.vercel.app"])

model = YOLO(MODEL_PATH)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return 'No image provided', 400

    file = request.files['image']
    image_bytes = file.read()
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Save temporarily
    os.makedirs("temp", exist_ok=True)
    temp_filename = f"temp/{uuid.uuid4().hex}.jpg"
    cv2.imwrite(temp_filename, img)

    # Run inference
    results = model(temp_filename, device='cpu')
    output_img = results[0].plot()

    # 🔍 Print detections and confidence
    print("\n--- Detection Results ---")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = results[0].names[cls_id]
        print(f"Detected {label} with {conf:.2%} confidence")
    print("-------------------------\n")

    # Convert and send image
    pil_img = Image.fromarray(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG')
    buf.seek(0)

    os.remove(temp_filename)
    return send_file(buf, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
