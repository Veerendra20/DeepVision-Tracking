# AI Surveillance: Human Tracking and Live Counting System

A complete end-to-end AI-based system for detecting, tracking, and counting humans in real-time or from static images.

## Features

- **Human Detection**: Powered by YOLOv8 for fast and accurate person detection.
- **Real-time Tracking**: Uses DeepSORT to maintain consistent IDs across frames.
- **Live & Total Counting**: Displays current persons in frame (Live Count) and cumulative unique individuals (Total Count).
- **Interactive UI**: Built with Streamlit for a seamless experience with Image Upload and Webcam modes.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd SURVAILANCE
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Linux/macOS
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

Launch the Streamlit application:
```bash
streamlit run app.py
```

## Project Structure

- `app.py`: Main Streamlit interface.
- `detection/`: Modules for YOLO human detection.
- `tracking/`: DeepSORT tracking integration.
- `utils/`: Counting and visualization helper functions.
- `models/`: Destination for saved model weights (e.g., `yolov8n.pt`).

## Requirements

- Python >= 3.9
- CUDA-compatible GPU (Optional, for better performance)
- Webcam (For real-time mode)

## License
MIT
