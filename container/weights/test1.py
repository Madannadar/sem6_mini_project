from ultralytics import YOLO
import cv2

MODEL_PATH = "best.pt"
IMAGE_PATH = r""

def detect_and_display(image_path):
    model = YOLO(MODEL_PATH)
    
    # Force CPU if needed
    results = model(image_path, device='cpu')

    # Visualize the results
    for result in results:
        # Plot boxes on the image
        img_with_boxes = result.plot()  # returns a numpy image with boxes and labels
        
        # Show the image using OpenCV
        cv2.imshow("Detected Defects", img_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

detect_and_display(IMAGE_PATH)
