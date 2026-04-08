import cv2

def draw_detections(frame, detections, color=(0, 255, 0), label="Person"):
    """
    Draw boxes for detections (without tracking).
    :param frame: Frame to draw on.
    :param detections: List of detections [[x1, y1, x2, y2, conf, cls], ...]
    :param color: BGR color for the box.
    :param label: Label for the box.
    """
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def draw_faces(frame, faces, color=(255, 0, 0)):
    """
    Draw boxes for detected faces.
    :param frame: Frame to draw on.
    :param faces: List of face bounding boxes [[x, y, w, h], ...]
    :param color: BGR color for the box.
    """
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
        cv2.putText(frame, "Face", (int(x), int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def draw_tracks(frame, tracks, color=(0, 255, 255)):
    """
    Draw boxes and IDs for tracked individuals.
    :param frame: Frame to draw on.
    :param tracks: List of active tracks from DeepSORT.
    :param color: BGR color for the box.
    """
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        ltrb = track.to_ltrb() # Left, Top, Right, Bottom
        
        cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), color, 2)
        cv2.putText(frame, f"ID: {track_id}", (int(ltrb[0]), int(ltrb[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def draw_count(frame, current_count, total_count):
    """
    Display both live and total counts on the frame.
    """
    # Draw a semi-transparent background for the counts
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Draw the counts
    cv2.putText(frame, f"LIVE COUNT: {current_count}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"TOTAL COUNT: {total_count}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return frame
