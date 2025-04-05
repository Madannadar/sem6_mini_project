from flask import Flask, request, send_file
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import io

app = Flask(__name__)
CORS(app)

model = YOLO("best.pt")

@app.route('/detect', methods=['POST'])
def detect():
    image_file = request.files['image']
    in_memory = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(in_memory, cv2.IMREAD_COLOR)

    results = model.predict(image)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"{model.names[cls]} {conf:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    _, buffer = cv2.imencode('.jpg', image)
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(port=5000)
