import cv2
from detection.person_detector import PersonDetector
from tracking.sort_tracker import SortTracker
from tracking.entry_exit import EntryExitCounter

# Initialize modules
detector = PersonDetector()
tracker = SortTracker()
counter = EntryExitCounter(line_y=250)  # Adjust line_y according to camera view

cap = cv2.VideoCapture(0)  # Camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1️⃣ Detect people
    detections = detector.detect(frame)

    # 2️⃣ Track with SORT
    tracks = tracker.update(detections)

    # 3️⃣ Update entry/exit counts
    counter.update(tracks)

    # 4️⃣ Draw bounding boxes and IDs
    for x1, y1, x2, y2, track_id in tracks:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {int(track_id)}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 5️⃣ Display counts
    entered, exited = counter.counts()
    cv2.putText(frame, f"IN: {entered}  OUT: {exited}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 6️⃣ Show video
    cv2.imshow("Passenger Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
