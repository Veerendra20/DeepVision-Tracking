class PeopleCounter:
    def __init__(self, persistence_threshold=5):
        """
        Initialize the people counter.
        :param persistence_threshold: Minimum frames a track must be confirmed before counting in total.
        """
        self.unique_ids = set()
        self.track_frame_counts = {} # track_id -> frame_count
        self.persistence_threshold = persistence_threshold

    def update(self, tracks):
        """
        Update the count of unique IDs and current attendees with temporal validation.
        :param tracks: List of active tracks from DeepSORT.
        :return: (current_count, total_unique_count)
        """
        current_count = 0
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            current_count += 1
            track_id = track.track_id
            
            # Initialize or increment frame count for this confirmed track
            if track_id not in self.track_frame_counts:
                self.track_frame_counts[track_id] = 0
            
            self.track_frame_counts[track_id] += 1
            
            # Only count in total if it has persisted for the threshold number of frames
            if self.track_frame_counts[track_id] >= self.persistence_threshold:
                self.unique_ids.add(track_id)
            
        return current_count, len(self.unique_ids)

    def get_count(self):
        """
        Get the total count of unique persons detected.
        """
        return len(self.unique_ids)

    def reset_count(self):
        """
        Reset the counter.
        """
        self.unique_ids.clear()
        self.track_frame_counts.clear()
