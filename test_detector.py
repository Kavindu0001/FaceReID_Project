import cv2

from utils.csv_logger import CSVLogger

from detection.person_detector import PersonDetector
from tracking.sort_tracker import SortTracker
from tracking.entry_exit import EntryExitCounter

from reid.face_extractor import FaceExtractor
from reid.face_embedder import FaceEmbedder
from reid.face_verifier import FaceVerifier


# ================= INITIALIZE =================
detector = PersonDetector()
tracker = SortTracker()
counter = EntryExitCounter(line_y=250)

face_extractor = FaceExtractor()
face_embedder = FaceEmbedder("reid_model.h5")
face_verifier = FaceVerifier(threshold=0.6)

logger = CSVLogger("passenger_log.csv")   # ✅ SINGLE LOGGER

cap = cv2.VideoCapture(0)


# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    tracks = tracker.update(detections)
    counter.update(tracks)

    for x1, y1, x2, y2, track_id in tracks:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        track_id = int(track_id)

        face = face_extractor.extract(frame, [x1, y1, x2, y2])
        label = ""

        if face is not None:
            embedding = face_embedder.get_embedding(face)

            # ===== ENTRY =====
            if counter.is_entering(track_id):
                face_verifier.register_entry(track_id, embedding)
                logger.log(track_id, "ENTER", True)
                label = "ENTER"

            # ===== EXIT =====
            elif counter.is_exiting(track_id):
                verified = face_verifier.verify_exit(track_id, embedding)
                logger.log(track_id, "EXIT", verified)
                label = "EXIT OK" if verified else "EXIT FAKE"

        color = (0, 255, 0) if label != "EXIT FAKE" else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID {track_id} {label}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    entered, exited = counter.counts()
    cv2.putText(
        frame,
        f"IN: {entered}  OUT: {exited}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.line(frame, (0, 250), (frame.shape[1], 250), (255, 255, 0), 2)

    cv2.imshow("Passenger Face-Verified Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()