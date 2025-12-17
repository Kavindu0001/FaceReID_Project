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
counter = EntryExitCounter(line_y=350)

face_extractor = FaceExtractor()
face_embedder = FaceEmbedder("reid_model.h5")
face_verifier = FaceVerifier(threshold=0.6)

logger = CSVLogger("passenger_log.csv")
cap = cv2.VideoCapture(0)

# ================= STATE =================
person_state = {}          # track_id -> "OUTSIDE" | "INSIDE"
logged_enter = set()       # IDs already logged ENTER
logged_exit = set()        # IDs already logged EXIT


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

        cy = (y1 + y2) // 2

        # 🔥 Initialize state based on position
        if track_id not in person_state:
            person_state[track_id] = (
                "INSIDE" if cy >= counter.line_y else "OUTSIDE"
            )

        label = ""
        color = (0, 255, 0)

        # 5️⃣ Face extraction
        face = face_extractor.extract(frame, [x1, y1, x2, y2])
        embedding = None
        if face is not None:
            embedding = face_embedder.get_embedding(face)

        # 🚧 STATE-LOCK PROTECTION (CRITICAL)
        if counter.is_entering(track_id) and person_state[track_id] == "INSIDE":
            pass
        elif counter.is_exiting(track_id) and person_state[track_id] == "OUTSIDE":
            pass

        # ===== ENTER EVENT (ONCE) =====
        elif (
            counter.is_entering(track_id)
            and track_id not in logged_enter
        ):
            verified = False
            if embedding is not None:
                face_verifier.register_entry(track_id, embedding)
                verified = True

            logger.log(track_id, "ENTER", verified)
            logged_enter.add(track_id)
            person_state[track_id] = "INSIDE"

            label = "ENTER"
            color = (0, 255, 0)

        # ===== EXIT EVENT (ONCE) =====
        elif (
            counter.is_exiting(track_id)
            and track_id not in logged_exit
        ):
            verified = False
            if embedding is not None:
                verified = face_verifier.verify_exit(track_id, embedding)

            logger.log(track_id, "EXIT", verified)
            logged_exit.add(track_id)
            person_state[track_id] = "OUTSIDE"

            label = "EXIT OK" if verified else "EXIT FAKE"
            color = (255, 0, 0) if verified else (0, 0, 255)

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

    # 6️⃣ Display counts
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

    # 7️⃣ Draw entry/exit line
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