import math

class PeopleCounter:
    def __init__(self, persistence_threshold=5, static_threshold=10, movement_min_px=5):
        """
        Initialize the people counter.
        :param persistence_threshold: Minimum frames a track must be confirmed before counting in total.
        :param static_threshold: Frames an object can stay static before it is ignored as a 'potential bag/object'.
        :param movement_min_px: Minimum pixels moved to be considered non-static.
        """
        self.unique_ids = set()
        self.track_frame_counts = {} # track_id -> frame_count
        self.persistence_threshold = persistence_threshold
        
        # Movement filtering
        self.prev_positions = {} # {track_id: (center_x, center_y)}
        self.static_frames = {} # {track_id: count}
        self.static_threshold = static_threshold
        self.movement_min_px = movement_min_px

    def update(self, tracks):
        """
        Update with temporal validation and movement-based filtering.
        :param tracks: List of active tracks from DeepSORT.
        :return: (current_count, total_unique_count)
        """
        current_count = 0
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb() # Left, Top, Right, Bottom
            center = ((ltrb[0] + ltrb[2]) / 2, (ltrb[1] + ltrb[3]) / 2)
            
            # Calculate movement distance
            if track_id in self.prev_positions:
                dist = math.sqrt((center[0] - self.prev_positions[track_id][0])**2 + 
                                 (center[1] - self.prev_positions[track_id][1])**2)
                
                if dist < self.movement_min_px:
                    self.static_frames[track_id] = self.static_frames.get(track_id, 0) + 1
                else:
                    self.static_frames[track_id] = 0
            else:
                self.static_frames[track_id] = 0
            
            self.prev_positions[track_id] = center
            
            # Filter out static objects (like bags)
            if self.static_frames[track_id] > self.static_threshold:
                continue
            
            # If moving (or not yet confirmed as static), include in current count
            current_count += 1
            
            # Temporal validation for total count
            if track_id not in self.track_frame_counts:
                self.track_frame_counts[track_id] = 0
            
            self.track_frame_counts[track_id] += 1
            
            # Only add to total if it has persisted for the threshold number of frames
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
        self.prev_positions.clear()
        self.static_frames.clear()
