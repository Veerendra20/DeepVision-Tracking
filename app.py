import streamlit as st
import cv2
import PIL.Image
import numpy as np
import tempfile
import time
from detection.yolo_detector import YOLODetector
from detection.yolo_detector import YOLODetector
from detection.face_detector import FaceDetector
from tracking.tracker import PersonTracker
from utils.counting import PeopleCounter
from utils.visualization import draw_detections, draw_tracks, draw_count

# Page config
st.set_page_config(page_title="AI Surveillance System", page_icon="🛡️", layout="wide")

# Initialize models
@st.cache_resource
def load_models():
    yolo = YOLODetector(model_path='yolov8n.pt')
    face = FaceDetector()
    return yolo, face

yolo_detector, face_detector = load_models()

def main():
    st.title("🛡️ AI-based Human and Face Tracking System")
    st.markdown("""
    This system detects humans and faces, tracks individuals with unique IDs, and counts them in real-time.
    """)

    # Sidebar
    st.sidebar.header("Settings")
    mode = st.sidebar.radio("Select Mode", ["Image Upload", "Webcam (Real-time)"])
    
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    show_ids = st.sidebar.checkbox("Track IDs", value=True)

    if mode == "Image Upload":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Load image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            original_image = image.copy()
            
            # Process image
            with st.spinner("Processing..."):
                # Human Detection
                detections = yolo_detector.detect(image, conf_threshold=conf_threshold)
                
                # Draw
                processed_image = image.copy()
                processed_image = draw_detections(processed_image, detections, face_detector=face_detector)
                
                # In Image Upload mode, Live and Total count are the same as it's a single frame
                processed_image = draw_count(processed_image, len(detections), len(detections))
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.subheader("Processed Image")
                st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            st.success(f"Detected {len(detections)} person(s).")

    elif mode == "Webcam (Real-time)":
        st.subheader("Webcam Live Feed")
        run = st.checkbox("Run Webcam")
        
        # Initialize Tracker and Counter
        tracker = PersonTracker()
        counter = PeopleCounter()
        
        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not access webcam. Please ensure it is connected and accessible.")
            run = False

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image")
                break
            
            # Detect
            detections = yolo_detector.detect(frame, conf_threshold=conf_threshold)
            
            # Track
            tracks = tracker.update(detections, frame)
            
            # Count
            live_count, total_count = counter.update(tracks)
            
            # Visualization
            processed_frame = frame.copy()
            if show_ids:
                processed_frame = draw_tracks(processed_frame, tracks, face_detector=face_detector)
            else:
                processed_frame = draw_detections(processed_frame, detections, face_detector=face_detector)
            
            processed_frame = draw_count(processed_frame, live_count, total_count)
            
            # Convert BGR to RGB
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(processed_frame)
            
            # Small delay to keep UI responsive
            time.sleep(0.01)
            
        else:
            cap.release()
            st.info("Webcam stopped.")

if __name__ == "__main__":
    main()
