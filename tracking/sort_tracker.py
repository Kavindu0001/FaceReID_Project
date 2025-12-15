# tracking/sort_tracker.py

import sys
import numpy as np

sys.path.append("sort")
from sort import Sort


class SortTracker:
    def __init__(self):
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    def update(self, detections):
        """
        detections: list of [x1, y1, x2, y2, confidence]
        """
        if len(detections) == 0:
            return []

        dets = np.array(detections)
        tracks = self.tracker.update(dets)
        return tracks
