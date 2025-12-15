# counting/line_counter.py

class LineCounter:
    def __init__(self, line_y):
        """
        line_y: Y coordinate of the counting line
        """
        self.line_y = line_y
        self.count_in = 0
        self.count_out = 0
        self.track_positions = {}  # track_id -> last y position

    def update(self, tracks):
        """
        tracks: ndarray of [x1, y1, x2, y2, track_id]
        """
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            cy = int((y1 + y2) / 2)

            if track_id not in self.track_positions:
                self.track_positions[track_id] = cy
                continue

            prev_cy = self.track_positions[track_id]

            # Crossing logic
            if prev_cy < self.line_y and cy >= self.line_y:
                self.count_in += 1
                print(f"IN  | ID {int(track_id)}")

            elif prev_cy > self.line_y and cy <= self.line_y:
                self.count_out += 1
                print(f"OUT | ID {int(track_id)}")

            self.track_positions[track_id] = cy
