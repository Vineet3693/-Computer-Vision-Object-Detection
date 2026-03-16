"""
Configuration settings for the Object Detection System
All settings centralized in one place
"""

# ============================================================================
# MODEL SELECTION & CONFIGURATION
# ============================================================================

# Available YOLOv8 Models (from nano to xlarge)
AVAILABLE_MODELS = {
    'yolov8n': {
        'name': 'YOLOv8 Nano',
        'size': '3.3MB',
        'speed': 'Fastest',
        'accuracy': 'Lower',
        'params': '3.2M',
        'use_case': 'Real-time detection on edge devices, low-latency webcam'
    },
    'yolov8s': {
        'name': 'YOLOv8 Small',
        'size': '11.2MB',
        'speed': 'Fast',
        'accuracy': 'Good',
        'params': '11.2M',
        'use_case': 'Balanced accuracy/speed, recommended for webcam'
    },
    'yolov8m': {
        'name': 'YOLOv8 Medium',
        'size': '26.4MB',
        'speed': 'Medium',
        'accuracy': 'High',
        'params': '25.9M',
        'use_case': 'High accuracy with reasonable speed, good for videos'
    },
    'yolov8l': {
        'name': 'YOLOv8 Large',
        'size': '52.9MB',
        'speed': 'Slow',
        'accuracy': 'Very High',
        'params': '43.7M',
        'use_case': 'Maximum accuracy, best results, slower inference'
    },
    'yolov8x': {
        'name': 'YOLOv8 XL',
        'size': '84.9MB',
        'speed': 'Very Slow',
        'accuracy': 'Best',
        'params': '68.2M',
        'use_case': 'Highest accuracy, production-grade detection'
    }
}

# Default Model Selection
DEFAULT_MODEL = 'yolov8s'  # Change to: yolov8n, yolov8m, yolov8l, or yolov8x
MODEL_PATH = f"models/{DEFAULT_MODEL}.pt"  # Auto-downloaded if not exists
MODEL_NAME = DEFAULT_MODEL

# ============================================================================
# RECOMMENDED THRESHOLDS BY MODEL
# ============================================================================

MODEL_THRESHOLDS = {
    'yolov8n': {
        'confidence': 0.7,   # Higher threshold needed for nano model
        'iou': 0.55,
        'description': 'Speed-optimized settings'
    },
    'yolov8s': {
        'confidence': 0.65,  # Balanced settings
        'iou': 0.5,
        'description': 'Recommended settings (current)'
    },
    'yolov8m': {
        'confidence': 0.6,   # Can be lower with medium model
        'iou': 0.45,
        'description': 'Accuracy-optimized settings'
    },
    'yolov8l': {
        'confidence': 0.55,  # More confident predictions
        'iou': 0.45,
        'description': 'High accuracy settings'
    },
    'yolov8x': {
        'confidence': 0.5,   # Nano has highest confidence
        'iou': 0.45,
        'description': 'Maximum accuracy settings'
    }
}

# Get thresholds for current model
_model_config = MODEL_THRESHOLDS.get(DEFAULT_MODEL, MODEL_THRESHOLDS['yolov8s'])

# Detection Thresholds
CONFIDENCE_THRESHOLD = _model_config['confidence']  # Minimum confidence for detection
IOU_THRESHOLD = _model_config['iou']  # IoU threshold for NMS

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

# Alert Settings
ALERT_ON_PERSON_DETECTION = False  # Enable alerts when persons are detected
PERSON_COUNT_THRESHOLD = 5  # Threshold for person count alerts

# Tracking Configuration (for ByteTrack)
TRACKING_ENABLED = True
TRACK_BUFFER = 30  # Frames to keep track of lost objects

# Performance Metrics
TARGET_FPS = 20
MAX_INFERENCE_TIME_MS = 50

# Alert Configuration
ALERT_ON_PERSON_DETECTION = False
PERSON_COUNT_THRESHOLD = 5  # Trigger alert if more than this many persons detected
