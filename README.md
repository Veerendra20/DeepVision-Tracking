# 🛡️ Professional AI Surveillance Dashboard

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLOv8](https://img.shields.io/badge/Object%20Detection-YOLOv8-green)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-yellow)

A high-performance Human Tracking & Analytics system built with YOLOv8, DeepSORT, and Streamlit. This system provides real-time head-level tracking, cumulative counting with temporal validation, and live analytics.

## 🌟 Key Features

- **🎯 Human-Centric Face Tracking**: Automatically identifies individuals and focuses bounding boxes on the face area for a professional, targeted interface.
- **📼 Video File Processing**: Integrated support for analyzing recorded footage (`.mp4`, `.avi`, `.mov`) with full tracking persistence.
- **🔢 Dual-Metric Counting**:
  - **Live Count**: Real-time occupancy metrics.
  - **Total Unique Count**: Cumulative tally of unique individuals validated through temporal persistence.
- **🛡️ Multi-Layer Validation**:
  - **Temporal Persistence**: Requires 5 frames (configurable) of confirmation to eliminate detection flicker.
  - **Movement Filtering**: Distinguishes between moving humans and static objects using a 5px/10-frame displacement algorithm.
- **📊 Professional Analytics**:
  - Real-time **Line Charts** for historical occupancy tracking.
  - Metric cards for **Live FPS** and detection analytics.
- **📄 Audit Logging**: Automatic background logging of detection events to `surveillance.log`.

## 🛠️ Technology Stack

- **Core Engine**: Python 3.9+
- **Detection**: YOLOv8 (Ultralytics)
- **Tracking**: DeepSORT (deepsort-realtime)
- **UI/Dashboard**: Streamlit
- **Data Handling**: Pandas & NumPy
- **Image Processing**: OpenCV (cv2)

## 📂 Project Structure

- `app.py`: Main Dashboard and UI Logic.
- `config.py`: Global Surveillance Parameters.
- `detection/`: Object and face detection modules.
- `tracking/`: DeepSORT tracker implementation.
- `utils/`: Core logic for counting, validation, and visualization.

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```

## ⚙️ Configuration

Fine-tune the system in `config.py`:
- `PERSISTENCE_THRESHOLD`: Minimum frames for "Confirmed" status.
- `MOVEMENT_MIN_PX`: Displacement threshold for movement detection.
- `STATIC_THRESHOLD`: Grace period before an object is flagged as "Static".

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed for high-accuracy human surveillance and occupancy analytics.*
