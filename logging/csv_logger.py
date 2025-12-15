import csv
import os
from datetime import datetime

class CSVLogger:
    def __init__(self, file_path="logs/entry_exit_log.csv"):
        self.file_path = file_path
        self._create_file_if_needed()

    def _create_file_if_needed(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "track_id",
                    "event",
                    "face_verified"
                ])

    def log(self, track_id, event, face_verified):
        with open(self.file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                track_id,
                event,
                face_verified
            ])
