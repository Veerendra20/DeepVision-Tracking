import streamlit as st
import cv2
import PIL.Image
import numpy as np
import pandas as pd
import time
import config
from detection.yolo_detector import YOLODetector
from detection.face_detector import FaceDetector
from tracking.tracker import PersonTracker
from utils.counting import PeopleCounter
from utils.visualization import draw_detections, draw_tracks, draw_count

# Page config for professional appearance
st.set_page_config(page_title="AI Surveillance Dashboard", page_icon="🛡️", layout="wide")

# Custom CSS for polished look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4149;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize models
@st.cache_resource
def load_models():
    yolo = YOLODetector(model_path=config.YOLO_MODEL)
    face = FaceDetector()
    return yolo, face

yolo_detector, face_detector = load_models()

def main():
    st.title("🛡️ Professional AI Surveillance Dashboard")
    st.markdown("---")

    # Sidebar Organization
    st.sidebar.header("🛠️ System Configuration")
    
    with st.sidebar.expander("🔍 Detection Settings", expanded=True):
        mode = st.radio("Processing Mode", ["Real-time Webcam", "Static Image Upload"])
        conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, config.DEFAULT_CONFIDENCE, 0.05)
        show_ids = st.checkbox("Enable Tracking IDs", value=True)

    with st.sidebar.expander("📈 Analytics Settings"):
        show_chart = st.checkbox("Show Live Analytics Chart", value=True)
        chart_size = st.number_input("Chart History (frames)", 10, 500, config.CHART_HISTORY_SIZE)

    if st.sidebar.button("🔄 Reset Global Session", type="primary"):
        st.cache_resource.clear()
        st.rerun()

    # Main UI Logic
    if mode == "Static Image Upload":
        uploaded_file = st.file_uploader("Upload Security Frame (JPG/PNG)", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            with st.spinner("Analyzing high-resolution frame..."):
                detections = yolo_detector.detect(image, conf_threshold=conf_threshold)
                processed_image = draw_detections(image.copy(), detections, face_detector=face_detector)
                processed_image = draw_count(processed_image, len(detections), len(detections))
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Capture")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.subheader("Face-Localized Analysis")
                st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            st.success(f"Detections Completed: {len(detections)} individuals identified.")

    else:
        # Webcam Mode with Advanced Analytics
        col_vid, col_stats = st.columns([2, 1])
        
        with col_vid:
            st.subheader("📹 Live Surveillance Feed")
            run = st.checkbox("Activate Camera System", key="cam_switch")
            FRAME_WINDOW = st.image([])
            
        with col_stats:
            st.subheader("📊 Live Analytics")
            m1, m2, m3 = st.columns(3)
            curr_metric = m1.empty()
            total_metric = m2.empty()
            fps_metric = m3.empty()
            
            chart_container = st.empty()
            log_container = st.empty()

        # Initialize Logic Components
        tracker = PersonTracker()
        counter = PeopleCounter()
        
        # Analytics state
        count_history = []
        fps_history = []
        
        cap = cv2.VideoCapture(0)
        
        prev_time = time.time()
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Access denied: Peripheral camera hardware not found.")
                break
            
            # Processing Pipeline
            detections = yolo_detector.detect(frame, conf_threshold=conf_threshold)
            tracks = tracker.update(detections, frame)
            live_count, total_count = counter.update(tracks)
            
            # Performance Calc
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            fps_history.append(fps)
            if len(fps_history) > 30: fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # Visualization
            processed_frame = frame.copy()
            if show_ids:
                processed_frame = draw_tracks(processed_frame, tracks, face_detector=face_detector)
            else:
                processed_frame = draw_detections(processed_frame, detections, face_detector=face_detector)
            
            processed_frame = draw_count(processed_frame, live_count, total_count)
            
            # Update Dashboard
            FRAME_WINDOW.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            
            curr_metric.metric("Live", live_count)
            total_metric.metric("Total", total_count)
            fps_metric.metric("FPS", f"{avg_fps:.1f}")
            
            # Chart update
            if show_chart:
                count_history.append(live_count)
                if len(count_history) > chart_size: count_history.pop(0)
                chart_container.line_chart(pd.DataFrame(count_history, columns=["Live Occupancy"]), height=200)

            # Session Log update (Small sampling)
            if total_count > 0:
                log_container.info(f"System Status: Operational. Total Unique Identified: {total_count}")

            time.sleep(0.01)
            
        else:
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            st.info("System Standby: Security protocols inactive.")

if __name__ == "__main__":
    main()
