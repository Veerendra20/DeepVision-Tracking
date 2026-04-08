import cv2
from typing import List, Tuple, Any

def draw_detections(frame: Any, detections: List, face_detector: Any = None, color: Tuple[int, int, int] = (0, 255, 0), label: str = "Person") -> Any:
    """
    Draw boxes for detections, focusing on the face area if possible.
    """
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        
        # Default face area heuristic (top 25% of box)
        h = y2 - y1
        face_box = [int(x1), int(y1), int(x2), int(y1 + (h * 0.25))]
        
        if face_detector:
            roi_y2 = int(y1 + (h * 0.5))
            roi = frame[int(y1):roi_y2, int(x1):int(x2)]
            if roi.size > 0:
                faces = face_detector.detect(roi)
                if len(faces) > 0:
                    fx, fy, fw, fh = faces[0]
                    face_box = [int(x1 + fx), int(y1 + fy), int(x1 + fx + fw), int(y1 + fy + fh)]

        cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (face_box[0], face_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def draw_tracks(frame: Any, tracks: List, face_detector: Any = None, color: Tuple[int, int, int] = (0, 255, 255)) -> Any:
    """
    Draw boxes and IDs for tracked individuals, prioritizing the face area.
    """
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        ltrb = track.to_ltrb()
        
        # Default face area heuristic
        h = ltrb[3] - ltrb[1]
        face_box = [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[1] + (h * 0.25))]
        
        if face_detector:
            roi_y2 = int(ltrb[1] + (h * 0.5))
            roi = frame[int(ltrb[1]):roi_y2, int(ltrb[0]):int(ltrb[2])]
            if roi.size > 0:
                faces = face_detector.detect(roi)
                if len(faces) > 0:
                    fx, fy, fw, fh = faces[0]
                    face_box = [int(ltrb[0] + fx), int(ltrb[1] + fy), int(ltrb[0] + fx + fw), int(ltrb[1] + fy + fh)]

        cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), color, 2)
        cv2.putText(frame, f"ID: {track_id}", (face_box[0], face_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def draw_count(frame: Any, current_count: int, total_count: int) -> Any:
    """
    Draw current and total attendee counts on the frame.
    """
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (250, 100), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    cv2.putText(frame, f"LIVE: {current_count}", (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"TOTAL: {total_count}", (20, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return frame
