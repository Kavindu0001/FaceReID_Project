import cv2

from detection.person_detector import PersonDetector
from tracking.sort_tracker import SortTracker
from tracking.entry_exit import EntryExitCounter

from reid.face_extractor import FaceExtractor
from reid.face_embedder import FaceEmbedder
from reid.face_verifier import FaceVerifier


# ================= INITIALIZE MODULES =================
detector = PersonDetector()
tracker = SortTracker()
counter = EntryExitCounter(line_y=250)

face_extractor = FaceExtractor()
face_embedder = FaceEmbedder("reid_model.h5")
face_verifier = FaceVerifier(threshold=0.6)

cap = cv2.VideoCapture(0)


# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1️⃣ Detect people
    detections = detector.detect(frame)

    # 2️⃣ Track with SORT
    tracks = tracker.update(detections)

    # 3️⃣ Update entry / exit logic
    counter.update(tracks)

    # 4️⃣ Process each tracked person
    for x1, y1, x2, y2, track_id in tracks:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        face = face_extractor.extract(frame, [x1, y1, x2, y2])
        label = ""

        if face is not None:
            embedding = face_embedder.get_embedding(face)

            if counter.is_entering(track_id):
                face_verifier.register_entry(track_id, embedding)
                label = "ENTER"

            elif counter.is_exiting(track_id):
                verified = face_verifier.verify_exit(track_id, embedding)
                label = "EXIT OK" if verified else "EXIT FAKE"

        color = (0, 255, 0) if label != "EXIT FAKE" else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID {int(track_id)} {label}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # 5️⃣ Show counts
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

    # Draw virtual door line
    cv2.line(frame, (0, 250), (frame.shape[1], 250), (255, 255, 0), 2)

    cv2.imshow("Passenger Face-Verified Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()