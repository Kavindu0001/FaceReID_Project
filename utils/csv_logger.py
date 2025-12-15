import csv
import os
from datetime import datetime


class CSVLogger:
    def __init__(self, filename="passenger_log.csv"):
        self.filename = filename

        # Check if file already exists
        file_exists = os.path.isfile(self.filename)

        # Open file in append mode
        self.file = open(self.filename, mode="a", newline="")
        self.writer = csv.writer(self.file)

        # Write header only once
        if not file_exists:
            self.writer.writerow([
                "timestamp",
                "track_id",
                "event",
                "face_verified"
            ])
            self.file.flush()

    def log(self, track_id, event, verified):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.writer.writerow([
            timestamp,
            int(track_id),
            event,
            bool(verified)
        ])

        # Force write to disk immediately
        self.file.flush()

        # Debug print (KEEP while testing)
        print(f"[CSV] {timestamp} | ID={track_id} | {event} | verified={verified}")

    def close(self):
        if self.file:
            self.file.close()