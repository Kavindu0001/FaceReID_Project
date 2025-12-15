import numpy as np

class FaceVerifier:
    def __init__(self, threshold=0.6):
        self.threshold = threshold
        self.database = {}  # track_id → embedding

    def register_entry(self, track_id, embedding):
        if track_id not in self.database:
            self.database[track_id] = embedding

    def verify_exit(self, track_id, embedding):
        if track_id not in self.database:
            return False

        stored = self.database[track_id]
        dist = np.linalg.norm(stored - embedding)

        return dist < self.threshold