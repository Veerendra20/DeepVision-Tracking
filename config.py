# Surveillance System Configuration

# Model Settings
YOLO_MODEL = 'yolov8n.pt'
DEFAULT_CONFIDENCE = 0.5

# Tracking & Counting Thresholds
# Minimum frames a track must be confirmed before adding to total count
PERSISTENCE_THRESHOLD = 5

# Movement distance (pixels) below which a track is considered 'static'
MOVEMENT_MIN_PX = 5

# Total consecutive static frames after which a track is ignored
STATIC_THRESHOLD = 10

# UI Settings
CHART_HISTORY_SIZE = 100
LOG_FILE = "surveillance.log"
