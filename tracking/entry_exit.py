class EntryExitCounter:
    def __init__(self, line_y):
        self.line_y = line_y
        self.track_last_y = {}

        self.entered_ids = set()
        self.exited_ids = set()

        # 🔹 per-frame flags
        self._enter_now = set()
        self._exit_now = set()

    def update(self, tracks):
        """
        tracks: [x1, y1, x2, y2, track_id]
        """
        # reset frame-level events
        self._enter_now.clear()
        self._exit_now.clear()

        for x1, y1, x2, y2, track_id in tracks:
            track_id = int(track_id)
            cy = int((y1 + y2) / 2)

            if track_id not in self.track_last_y:
                self.track_last_y[track_id] = cy
                continue

            prev_y = self.track_last_y[track_id]

            # 🚶 ENTER (top → bottom)
            if prev_y < self.line_y and cy >= self.line_y:
                self.entered_ids.add(track_id)
                self._enter_now.add(track_id)

            # 🚶 EXIT (bottom → top)
            if prev_y > self.line_y and cy <= self.line_y:
                self.exited_ids.add(track_id)
                self._exit_now.add(track_id)

            self.track_last_y[track_id] = cy

    # ✅ USED BY test_detector.py
    def is_entering(self, track_id):
        return int(track_id) in self._enter_now

    def is_exiting(self, track_id):
        return int(track_id) in self._exit_now

    def counts(self):
        return len(self.entered_ids), len(self.exited_ids)