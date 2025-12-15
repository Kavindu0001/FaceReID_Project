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

logger = CSVLogger("passenger_log.csv")

cap = cv2.VideoCapture(0)


# ================= STATE =================
person_state = {}  # track_id -> "OUTSIDE" | "INSIDE"


# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1️⃣ Detect persons
    detections = detector.detect(frame)

    # 2️⃣ Track persons
    tracks = tracker.update(detections)

    # 3️⃣ Update entry / exit counter
    counter.update(tracks)

    # 4️⃣ Process each tracked person
    for x1, y1, x2, y2, track_id in tracks:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        track_id = int(track_id)

        # Init state
        if track_id not in person_state:
            person_state[track_id] = "OUTSIDE"

        label = ""
        color = (0, 255, 0)

        # Face extraction
        face = face_extractor.extract(frame, [x1, y1, x2, y2])
        embedding = None
        if face is not None:
            embedding = face_embedder.get_embedding(face)

        cy = (y1 + y2) // 2

        # 🔍 DEBUG (VERY IMPORTANT)
        print(
            f"ID={track_id}, cy={cy}, "
            f"state={person_state[track_id]}, "
            f"enter={counter.is_entering(track_id)}, "
            f"exit={counter.is_exiting(track_id)}"
        )

        # ===== ENTER =====
        if (
            counter.is_entering(track_id)
            and person_state[track_id] == "OUTSIDE"
        ):
            verified = embedding is not None
            if verified:
                face_verifier.register_entry(track_id, embedding)

            logger.log(track_id, "ENTER", verified)
            person_state[track_id] = "INSIDE"
            label = "ENTER"

        # ===== EXIT =====
        elif (
            counter.is_exiting(track_id)
            and person_state[track_id] == "INSIDE"
        ):
            verified = False
            if embedding is not None:
                verified = face_verifier.verify_exit(track_id, embedding)

            logger.log(track_id, "EXIT", verified)
            person_state[track_id] = "OUTSIDE"
            label = "EXIT OK" if verified else "EXIT FAKE"
            color = (0, 0, 255) if not verified else (255, 0, 0)

        # 🎨 Draw bounding box
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

    # 5️⃣ Display counts
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

    # 6️⃣ Draw entry/exit line
    cv2.line(
        frame,
        (0, counter.line_y),
        (frame.shape[1], counter.line_y),
        (0, 255, 255),
        2
    )

    cv2.imshow("Passenger Face-Verified Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
logger.close()