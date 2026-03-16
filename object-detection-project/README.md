# 🎯 AI Object Detection System

A complete, production-ready computer vision system for real-time object detection using YOLOv8, OpenCV, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-orange.svg)

---

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Dashboard](#dashboard)
- [Performance Metrics](#performance-metrics)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## ✨ Features

### Core Capabilities
- 🔍 **Multi-Object Detection**: Detect 80 COCO classes including persons, vehicles, animals, and everyday objects
- 📷 **Multiple Input Sources**: Support for images, videos, and real-time webcam feed
- 🎨 **Color-Coded Visualization**: Different colors for different object categories
- 📊 **Real-Time Statistics**: Live FPS, detection counts, and confidence metrics
- 🔔 **Alert System**: Configurable alerts for threshold-based detections
- 📝 **Logging & Export**: CSV logging of all detections with bounding box coordinates

### Advanced Features
- 🎯 **Object Tracking**: Track objects across video frames with unique IDs (ByteTrack support)
- ⚡ **GPU Acceleration**: Automatic CUDA/MPS detection for faster inference
- 🎛️ **Interactive Dashboard**: Streamlit-based UI for easy interaction
- 📈 **Performance Metrics**: Precision, recall, FPS, and inference time tracking
- 🔧 **Configurable Thresholds**: Adjustable confidence and IoU thresholds

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────┐
│              INPUT LAYER                 │
│   Image / Video / Webcam                │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│           PREPROCESSING                  │
│   Resize → Normalize → BGR2RGB          │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│            MODEL LAYER                   │
│   YOLOv8 (Backbone → Neck → Head)       │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│          DETECTION LAYER                 │
│   Confidence Filter → NMS               │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│         POST PROCESSING                  │
│   Count → Categorize → Alert → Log      │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│            OUTPUT LAYER                  │
│   Annotated Feed / Saved Files / UI     │
└─────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- GPU with CUDA (optional, for faster inference)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd object-detection-project
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "from ultralytics import YOLO; print('✅ YOLOv8 installed successfully!')"
```

---

## 🚀 Quick Start

### Run Webcam Detection
```bash
python main.py --source webcam
```

### Detect Objects in an Image
```bash
python main.py --source inputs/images/test.jpg
```

### Process a Video File
```bash
python main.py --source inputs/videos/test.mp4
```

### Launch Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 💻 Usage

### Command-Line Interface

#### Basic Usage
```bash
# Webcam detection
python main.py --source webcam

# Image detection
python main.py --source path/to/image.jpg

# Video detection
python main.py --source path/to/video.mp4
```

#### Advanced Options
```bash
# Use a specific model
python main.py --source webcam --model yolov8m.pt

# Adjust confidence threshold
python main.py --source webcam --confidence 0.7

# Enable object tracking
python main.py --source video.mp4 --tracking

# Disable display window (headless mode)
python main.py --source video.mp4 --no-display
```

#### CLI Arguments Reference
| Argument | Description | Default |
|----------|-------------|---------|
| `--source` | Input source (image/video/webcam) | webcam |
| `--model` | YOLOv8 model variant | yolov8s.pt |
| `--confidence` | Detection confidence threshold | 0.5 |
| `--iou` | IoU threshold for NMS | 0.45 |
| `--tracking` | Enable object tracking | False |
| `--no-display` | Disable output window | False |

---

## 📁 Project Structure

```
object-detection-project/
│
├── inputs/                      # Input files directory
│   ├── images/                  # Test images
│   ├── videos/                  # Test videos
│   └── webcam/                  # Webcam (live)
│
├── models/                      # Model weights (auto-downloaded)
│   └── yolov8s.pt
│
├── outputs/                     # Output files
│   ├── detected_images/         # Processed images
│   ├── detected_videos/         # Processed videos
│   └── logs/                    # Detection logs (CSV)
│
├── src/                         # Source code
│   ├── config.py               # Configuration settings
│   ├── preprocess.py           # Input preprocessing
│   ├── detect.py               # Core detection engine
│   ├── postprocess.py          # Result processing
│   ├── tracker.py              # Object tracking
│   └── visualize.py            # Visualization utilities
│
├── dashboard/                   # Streamlit UI
│   └── app.py                  # Dashboard application
│
├── utils/                       # Utilities
│   ├── logger.py               # Logging system
│   └── metrics.py              # Performance metrics
│
├── main.py                      # Entry point
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## ⚙️ Configuration

Edit `src/config.py` to customize settings:

```python
# Model Configuration
MODEL_PATH = "models/yolov8s.pt"
MODEL_NAME = "yolov8s"  # n, s, m, l, x

# Detection Thresholds
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Input Configuration
INPUT_SIZE = (640, 640)

# Class Colors (BGR format)
CLASS_COLORS = {
    'person': (0, 255, 0),      # Green
    'vehicle': (255, 0, 0),     # Blue
    'animal': (0, 255, 255),    # Yellow
    'object': (0, 0, 255),      # Red
}

# Alert Configuration
ALERT_ON_PERSON_DETECTION = False
PERSON_COUNT_THRESHOLD = 5
```

---

## 📖 API Reference

### DetectionEngine

```python
from src.detect import DetectionEngine

# Initialize
engine = DetectionEngine(
    model_path="yolov8s.pt",
    confidence_threshold=0.5,
    iou_threshold=0.45,
)

# Load model
engine.load_model()

# Run detection
detections = engine.detect(frame)

# Detection result format
{
    'bbox': [x1, y1, x2, y2],
    'class_id': 0,
    'class_name': 'person',
    'confidence': 0.95,
}
```

### PostProcessor

```python
from src.postprocess import PostProcessor

processor = PostProcessor()
result = processor.process(detections, source_type='webcam')

# Result contains:
# - stats: Detection statistics
# - alerts: Any triggered alerts
# - categorized: Detections by category
```

### Visualizer

```python
from src.visualize import Visualizer

visualizer = Visualizer()
annotated_frame = visualizer.visualize(
    frame,
    detections,
    stats=stats,
    fps=30.5,
    alerts=alerts,
)
```

---

## 🎨 Dashboard

The Streamlit dashboard provides an interactive interface for object detection.

### Launch Dashboard
```bash
streamlit run dashboard/app.py
```

### Features
- 📤 Upload images and videos
- 🎛️ Adjust detection parameters
- 📊 View detection statistics
- 📥 Download processed outputs
- 🎯 Select different YOLOv8 models

---

## 📈 Performance Metrics

The system tracks various performance metrics:

| Metric | Description | Target |
|--------|-------------|--------|
| **FPS** | Frames per second | >20 |
| **Inference Time** | Time per frame | <50ms |
| **Precision** | Correct detections | >80% |
| **Recall** | Objects found | >75% |
| **mAP@50** | Mean average precision | >50% |

---

## 📸 Examples

### Person Detection
```bash
python main.py --source webcam --confidence 0.6
```

### Vehicle Counting
```python
from src.config import VEHICLE_CLASSES
from src.detect import DetectionEngine

engine = DetectionEngine()
engine.load_model()

# Filter for vehicles only
detections = engine.detect(frame, classes=[2, 3, 5, 7])  # car, motorcycle, bus, truck
```

### Custom Alert System
```python
from src.postprocess import check_alerts

stats = calculate_statistics(detections)
alerts = check_alerts(stats)

for alert in alerts:
    print(f"⚠️ ALERT: {alert['message']}")
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Not Downloading
```bash
# Manual download
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
```

#### 2. CUDA Out of Memory
```bash
# Use smaller model
python main.py --source webcam --model yolov8n.pt
```

#### 3. Webcam Not Found
```bash
# Try different device index
python main.py --source 1  # or 2, 3, etc.
```

#### 4. Low FPS
- Use a smaller model (yolov8n or yolov8s)
- Reduce input resolution
- Enable GPU acceleration

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

Built with ❤️ using YOLOv8, OpenCV, and Streamlit
