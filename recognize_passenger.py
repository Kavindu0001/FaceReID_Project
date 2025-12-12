import cv2
import time
from utils.face_detector import FaceDetector
from utils.face_embedder import FaceEmbedder
from utils.database_manager import DatabaseManager
from utils.reid_matcher import ReidMatcher

# Initialize modules
detector = FaceDetector()
embedder = FaceEmbedder()
db = DatabaseManager()
matcher = ReidMatcher(threshold=0.6)

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
        passengers_db = db.load_passengers()
        passenger_id = matcher.find_best_match(embedding, passengers_db)
        if passenger_id is not None:
            db.log_event(passenger_id, "exit", time.strftime("%Y-%m-%d %H:%M:%S"))
            print(f"Passenger exiting: {passenger_id}")
        else:
            print("Passenger not found in database")

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
