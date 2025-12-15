import csv
import os
from datetime import datetime


class CSVLogger:
    def __init__(self, filename="passenger_log.csv"):
        self.filename = filename

        # Create file with header if not exists
        if not os.path.exists(self.filename):
            with open(self.filename, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "track_id", "event", "face_verified"])

    def log(self, track_id, event, verified):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, track_id, event, verified])