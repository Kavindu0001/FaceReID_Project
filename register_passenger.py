import cv2
import time
import uuid
from utils.face_detector import FaceDetector
from utils.face_embedder import FaceEmbedder
from utils.database_manager import DatabaseManager

# Initialize modules
detector = FaceDetector()
embedder = FaceEmbedder()
db = DatabaseManager()

# Open camera
cap = cv2.VideoCapture(0)  # Change to camera index if needed

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face = detector.detect_face(frame)
    if face is not None:
        embedding = embedder.get_embedding(face)
        passenger_id = str(uuid.uuid4())  # unique ID for passenger
        db.register_passenger(passenger_id, embedding)
        db.log_event(passenger_id, "entry", time.strftime("%Y-%m-%d %H:%M:%S"))
        print(f"Passenger registered: {passenger_id}")

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
