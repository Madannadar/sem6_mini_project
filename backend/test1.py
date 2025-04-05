import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Define model and image path
MODEL_PATH = "best.pt"  # Change to 'last.pt' if needed
IMAGE_PATH = r"D:\Coding\NextJS\Projects\sem6 mini project\test image.webp"

# Load the trained YOLOv8m model
model = YOLO(MODEL_PATH)

def detect_and_display(image_path):
    # Perform inference
    results = model(image_path)

    # Load the original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct display

    # Plot detected bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class ID
            label = f"{model.names[cls]} {conf:.2f}"  # Class name + confidence

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image with annotations
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# Run detection
detect_and_display(IMAGE_PATH)
