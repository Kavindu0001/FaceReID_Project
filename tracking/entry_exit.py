# tracking/entry_exit.py

class EntryExitCounter:
    def __init__(self, line_y):
        self.line_y = line_y          # y-coordinate of the “entry/exit line”
        self.entered = set()          # IDs that entered
        self.exited = set()           # IDs that exited
        self.track_positions = {}     # store previous center_y per track_id

    def update(self, tracks):
        """
        tracks: list of [x1, y1, x2, y2, track_id]
        """
        for x1, y1, x2, y2, track_id in tracks:
            cy = int((y1 + y2) / 2)  # center y of bounding box

            if track_id in self.track_positions:
                prev_cy = self.track_positions[track_id]

                # Detect ENTER (cross line downward)
                if prev_cy < self.line_y and cy > self.line_y:
                    self.entered.add(track_id)

                # Detect EXIT (cross line upward)
                if prev_cy > self.line_y and cy < self.line_y:
                    self.exited.add(track_id)

            # Update current position
            self.track_positions[track_id] = cy

    def counts(self):
        """Return total entered and exited counts"""
        return len(self.entered), len(self.exited)
