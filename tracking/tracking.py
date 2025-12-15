# tracking/tracking.py

class Tracker:
    def __init__(self):
        # Initialize your tracking data structures
        self.tracks = []

    def update(self, detections):
        """
        Update tracks with new detections.
        Args:
            detections (list): list of bounding boxes [x1, y1, x2, y2]
        """
        # TODO: implement tracking logic
        pass

    def get_tracks(self):
        return self.tracks
