import cv2
from mtcnn import MTCNN
from PIL import Image
import numpy as np

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()

    def detect_face(self, image):
        results = self.detector.detect_faces(image)
        if len(results) == 0:
            return None

        x, y, w, h = results[0]['box']
        face = image[y:y+h, x:x+w]
        return face
