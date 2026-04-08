# 🛡️ Professional AI Surveillance Dashboard

A high-performance Human Tracking & Analytics system built with YOLOv8, DeepSORT, and Streamlit. This system provides real-time head-level tracking, cumulative counting with temporal validation, and live analytics.

## 🌟 Key Features

- **🎯 Human-Centric Face Tracking**: Automatically identifies individuals and focuses bounding boxes on the face area.
- **📼 Video File Processing**: Integrated support for uploading and analyzing recorded footage (`.mp4`, `.avi`, `.mov`).
- **🔢 Dual-Metric Counting**:
  - **Live Count**: Real-time occupancy of the monitored area.
  - **Total Unique Count**: Cumulative tally of unique individuals detected during the session.
- **🛡️ Multi-Layer Validation**:
  - **Temporal Persistence**: Requires 5 frames of confirmation to filter out flicker.
  - **Movement Filtering**: Distinguishes between moving humans and static objects (bags, seats) using a 5px/10-frame displacement check.
- **📊 Analytics Dashboard**:
  - Real-time **Line Charts** for occupancy history.
  - Professional metrics including **Live FPS** and detection confidence.
  - **🔄 Session Reset**: Instantly clear history and logs.
- **📄 Event Logging**: Automatic background logging of detection events to `surveillance.log`.

## 📂 Project Structure

- `app.py`: Main Streamlit Dashboard.
- `config.py`: Centralized system parameters and thresholds.
- `detection/`: YOLOv8 and Face Detection modules.
- `tracking/`: DeepSORT tracking integration.
- `utils/`: Logic for counting, movement filtering, and advanced visualization.

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```

## 🛠️ Configuration

Adjust tracking sensitivity in `config.py`:
- `PERSISTENCE_THRESHOLD`: Frames required for confirmation.
- `MOVEMENT_MIN_PX`: Minimum movement to be considered "Living."
- `STATIC_THRESHOLD`: Frames allowed before an object is ignored as "Static."

## Requirements

- Python >= 3.9
- CUDA-compatible GPU (Optional, for better performance)
- Webcam (For real-time mode)

## License
MIT
