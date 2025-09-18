from ultralytics import YOLO

def detect_objects(image):
    model = YOLO("models/yolov8n.pt")
    results = model(image)
    return results

