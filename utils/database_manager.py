import json
import os
import numpy as np

class DatabaseManager:
    def __init__(self, db_path="database/passenger_store.json", log_path="database/entry_exit_log.json"):
        self.db_path = db_path
        self.log_path = log_path

        self._ensure_file(self.db_path, {"passengers": []})
        self._ensure_file(self.log_path, {"logs": []})

    def _ensure_file(self, path, default):
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump(default, f, indent=4)

    def load_passengers(self):
        with open(self.db_path, "r") as f:
            return json.load(f)

    def save_passengers(self, data):
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=4)

    def register_passenger(self, passenger_id, embedding):
        data = self.load_passengers()
        data["passengers"].append({
            "id": passenger_id,
            "embedding": embedding.tolist()
        })
        self.save_passengers(data)

    def log_event(self, passenger_id, event_type, timestamp):
        with open(self.log_path, "r") as f:
            logs = json.load(f)

        logs["logs"].append({
            "passenger_id": passenger_id,
            "event": event_type,
            "timestamp": timestamp
        })

        with open(self.log_path, "w") as f:
            json.dump(logs, f, indent=4)
