import math
import logging
from typing import List, Tuple, Dict, Set
import config

# Configure logging
logging.basicConfig(
    filename=config.LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class PeopleCounter:
    def __init__(self, 
                 persistence_threshold: int = config.PERSISTENCE_THRESHOLD, 
                 static_threshold: int = config.STATIC_THRESHOLD, 
                 movement_min_px: int = config.MOVEMENT_MIN_PX):
        """
        Initialize the professional people counter.
        """
        self.unique_ids: Set[int] = set()
        self.track_frame_counts: Dict[int, int] = {} 
        self.persistence_threshold = persistence_threshold
        
        # Movement filtering
        self.prev_positions: Dict[int, Tuple[float, float]] = {} 
        self.static_frames: Dict[int, int] = {} 
        self.static_threshold = static_threshold
        self.movement_min_px = movement_min_px
        
        logging.info("PeopleCounter initialized with persistence=%d, static_threshold=%d", 
                     persistence_threshold, static_threshold)

    def update(self, tracks: List) -> Tuple[int, int]:
        """
        Update counts with temporal validation and movement-based filtering.
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
            
            # If moving, include in current live count
            current_count += 1
            
            # Temporal validation for total cumulative count
            if track_id not in self.track_frame_counts:
                self.track_frame_counts[track_id] = 0
            
            self.track_frame_counts[track_id] += 1
            
            # Only add to total if it has persisted for the threshold number of frames
            if self.track_frame_counts[track_id] >= self.persistence_threshold:
                if track_id not in self.unique_ids:
                    self.unique_ids.add(track_id)
                    logging.info(f"[EVENT] New unique person detected. ID: {track_id}")
            
        return current_count, len(self.unique_ids)

    def get_count(self) -> int:
        """
        Get the total count of unique persons detected.
        """
        return len(self.unique_ids)

    def reset_count(self) -> None:
        """
        Reset all internal counters and logs for a new session.
        """
        self.unique_ids.clear()
        self.track_frame_counts.clear()
        self.prev_positions.clear()
        self.static_frames.clear()
        logging.info("PeopleCounter session reset.")
