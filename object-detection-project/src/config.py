"""
Configuration settings for the Object Detection System
All settings centralized in one place
"""

# Model Configuration
MODEL_PATH = "models/yolov8s.pt"  # Auto-downloaded if not exists
MODEL_NAME = "yolov8s"  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x

# Detection Thresholds
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detection
IOU_THRESHOLD = 0.45  # IoU threshold for NMS

# Input Configuration
INPUT_SIZE = (640, 640)  # Resize dimensions for model input

# Classes to Detect (COCO dataset has 80 classes)
# Common classes of interest
CLASSES_OF_INTEREST = [
    'person',      # class 0
    'bicycle',     # class 1
    'car',         # class 2
    'motorcycle',  # class 3
    'bus',         # class 5
    'truck',       # class 7
    'dog',         # class 17
    'cat',         # class 16
]

# All COCO classes (80 total)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Color mapping for different object classes
CLASS_COLORS = {
    'person': (0, 255, 0),      # Green
    'vehicle': (255, 0, 0),     # Blue (for car, bus, truck, etc.)
    'animal': (0, 255, 255),    # Yellow
    'object': (0, 0, 255),      # Red (default)
}

# Vehicle class grouping
VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']

# Animal class grouping
ANIMAL_CLASSES = ['dog', 'cat', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']

# Output Paths
OUTPUT_DIR = "outputs"
DETECTED_IMAGES_DIR = "outputs/detected_images"
DETECTED_VIDEOS_DIR = "outputs/detected_videos"
LOGS_DIR = "outputs/logs"
DETECTION_LOG_FILE = "outputs/logs/detection_log.csv"

# Webcam Configuration
WEBCAM_SOURCE = 0  # Default webcam
WEBCAM_WIDTH = 1280
WEBCAM_HEIGHT = 720

# Video Processing
VIDEO_FPS = 30
VIDEO_CODEC = "mp4v"

# Display Settings
SHOW_CONFIDENCE = True
SHOW_CLASS_LABEL = True
SHOW_BBOX = True
LINE_THICKNESS = 2
FONT_SCALE = 0.6

# Tracking Configuration (for ByteTrack)
TRACKING_ENABLED = True
TRACK_BUFFER = 30  # Frames to keep track of lost objects

# Performance Metrics
TARGET_FPS = 20
MAX_INFERENCE_TIME_MS = 50

# Alert Configuration
ALERT_ON_PERSON_DETECTION = False
PERSON_COUNT_THRESHOLD = 5  # Trigger alert if more than this many persons detected
