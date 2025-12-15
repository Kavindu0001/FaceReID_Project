from ultralytics import YOLO
import cv2

class PersonDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def detect(self, frame):
        results = self.model(frame, conf=0.5)
        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append([x1, y1, x2, y2])

        return detections
