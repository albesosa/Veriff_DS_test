from models.sort import Sort
import numpy as np


class FaceTracker:
    """
    Face tracking module using SORT (Simple Online and Realtime Tracking).
    This module provides a unified interface for face tracking using different methods.
    New tracking methods can be easily integrated by implementing the `update` method.
    Attributes:
        method (str): The tracking method to use ('sort').
        tracker: An instance of the selected tracking class.
    Methods:
      update(faces): Updates the tracker with new face detections and returns tracked faces.
    Usage:
        face_tracker = FaceTracker(method="sort")
        tracked_faces = face_tracker.update(faces)
    """
    def __init__(self, method="sort"):
        self.method = method
        if self.method == "sort":
            self.tracker = Sort()
        else:
            raise ValueError(f"Unsupported tracker method: {self.method}")


    def update(self, faces):
        """
        Takes a list of face bounding boxes (x, y, w, h)
        Returns list of (x, y, w, h, track_id)
        """
        detections = []
        for (x, y, w, h) in faces:
            x1, y1, x2, y2 = x, y, x + w, y + h
            detections.append([x1, y1, x2, y2, 1.0])  # Dummy confidence score

        detections = np.array(detections, dtype=np.float32) if detections else np.empty((0, 5), dtype=np.float32)
        tracks = self.tracker.update(detections)

        results = []
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            w, h = x2 - x1, y2 - y1
            x1, y1, w, h, track_id = int(x1), int(y1), int(w), int(h), int(track_id)
            results.append((x1, y1, w, h, track_id))

        return results
