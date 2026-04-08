class PeopleCounter:
    def __init__(self):
        """
        Initialize the people counter.
        """
        self.unique_ids = set()

    def update(self, tracks):
        """
        Update the count of unique IDs and current attendees.
        :param tracks: List of active tracks from DeepSORT.
        :return: (current_count, total_unique_count)
        """
        current_count = 0
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            current_count += 1
            track_id = track.track_id
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
