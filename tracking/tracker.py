from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from typing import List, Any

class PersonTracker:
    def __init__(self, max_age: int = 30, n_init: int = 3, nms_max_overlap: float = 1.0, max_cosine_distance: float = 0.2):
        """
        Initialize the DeepSORT tracker.
        :param max_age: Maximum number of frames to keep a track alive without detection.
        :param n_init: Number of consecutive detections before a track is confirmed.
        :param nms_max_overlap: Maximum overlap for Non-Maximum Suppression.
        :param max_cosine_distance: Threshold for cosine distance in appearance matching.
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=nms_max_overlap,
            max_cosine_distance=max_cosine_distance,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",  # Uses MobileNet for appearance embeddings
            half=True,
            bgr=True
        )

    def update(self, detections: List[List[float]], frame: np.ndarray) -> List[Any]:
        """
        Update tracks with new detections.
        :param detections: List of detections [[x1, y1, x2, y2, confidence, class_id], ...]
        :param frame: The current frame (for appearance embeddings).
        :return: List of active tracks.
        """
        # DeepSORT expects detections in the format [([x, y, w, h], confidence, class_id), ...]
        formatted_detections = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            w = x2 - x1
            h = y2 - y1
            formatted_detections.append(([x1, y1, w, h], conf, cls))

        tracks = self.tracker.update_tracks(formatted_detections, frame=frame)
        return tracks
