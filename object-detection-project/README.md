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
- 🏆 **5 YOLOv8 Models**: Choose from nano→xlarge based on accuracy/speed tradeoff
  - **YOLOv8 Nano**: Fastest (3.3MB)
  - **YOLOv8 Small**: Recommended balanced (11.2MB)  
  - **YOLOv8 Medium**: Higher accuracy (26.4MB)
  - **YOLOv8 Large**: Very high accuracy (52.9MB)
  - **YOLOv8 XL**: Maximum accuracy, production-grade (84.9MB)

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

### Real-Time Webcam Detection ✅
```bash
# Start real-time detection from your webcam
python main.py --source webcam

# With display window (default)
python main.py --source webcam --no-display=False

# Headless mode (processing without display)
python main.py --source webcam --no-display

# With higher accuracy (increased confidence threshold)
python main.py --source webcam --confidence 0.7

# With object tracking enabled
python main.py --source webcam --tracking
```

**Controls during webcam detection:**
- Press `q` to quit the application
- Press `s` to save a snapshot

### Detect Objects in an Image
```bash
python main.py --source inputs/images/test.jpg
```

### Process a Video File
```bash
python main.py --source inputs/videos/test.mp4
```

### Launch Interactive Dashboard
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

#### 🎯 Model Selection (For Better Accuracy!)

**5 YOLOv8 Models Available** - Choose based on your accuracy/speed needs:

```bash
# List all available models with details
python main.py --list-models

# Recommended for webcam (balanced accuracy/speed)
python main.py --source webcam --model-variant yolov8s

# Higher accuracy (slower)
python main.py --source webcam --model-variant yolov8m

# Very high accuracy (even slower)
python main.py --source webcam --model-variant yolov8l

# Best accuracy, production-grade, slowest
python main.py --source webcam --model-variant yolov8x

# Fastest, edge devices, lower accuracy
python main.py --source webcam --model-variant yolov8n
```

**Model Comparison:**

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| **YOLOv8 Nano** (n) | 3.3MB | Fastest ⚡ | Lower | Edge devices, real-time webcam |
| **YOLOv8 Small** (s) | 11.2MB | Fast ⚡⚡ | Good ✅ | **Recommended for webcam** |
| **YOLOv8 Medium** (m) | 26.4MB | Medium ⚡⚡⚡ | High ✅✅ | Better accuracy, slower |
| **YOLOv8 Large** (l) | 52.9MB | Slow ⚡⚡⚡⚡ | Very High ✅✅✅ | Video processing, best results |
| **YOLOv8 XL** (x) | 84.9MB | Very Slow ⚡⚡⚡⚡⚡ | Best ✅✅✅✅ | Production-grade detection |

#### Advanced Options
```bash
# Use medium model for better accuracy
python main.py --source webcam --model-variant yolov8m

# Use large model with maximum accuracy
python main.py --source video.mp4 --model-variant yolov8l --tracking

# Override confidence threshold
python main.py --source webcam --confidence 0.7

# Enable object tracking
python main.py --source video.mp4 --tracking

# Disable display window (headless mode)
python main.py --source video.mp4 --no-display

# Combine options for maximum accuracy
python main.py --source webcam --model-variant yolov8l --confidence 0.7 --iou 0.55 --tracking
```

#### CLI Arguments Reference
| Argument | Description | Default |
|----------|-------------|---------|
| `--source` | Input source (image/video path or 'webcam') | webcam |
| `--model-variant` | YOLOv8 variant (n/s/m/l/x) | yolov8s |
| `--model` | [Advanced] Full path to custom model weights | - |
| `--confidence` | Detection confidence threshold (0-1) | Model-specific* |
| `--iou` | IoU threshold for NMS (0-1) | Model-specific* |
| `--tracking` | Enable object tracking (ByteTrack) | False |
| `--no-display` | Disable output display window | False |
| `--list-models` | Show all available models and exit | - |

*Model-specific defaults: yolov8n (0.7), yolov8s (0.65), yolov8m (0.6), yolov8l (0.55), yolov8x (0.5)

#### Improving Accuracy - Complete Guide

If you're experiencing false positives or incorrect detections:

1. **Try a larger model (Best for accuracy):**
   ```bash
   # Small to medium (much better accuracy)
   python main.py --source webcam --model-variant yolov8m
   
   # Medium to large (even better)
   python main.py --source webcam --model-variant yolov8l
   ```

2. **Increase confidence threshold:**
   ```bash
   python main.py --source webcam --confidence 0.75  # Higher = fewer but more confident detections
   ```

3. **Adjust IOU threshold to reduce overlapping detections:**
   ```bash
   python main.py --source webcam --iou 0.55  # Higher = fewer overlapping boxes
   ```

4. **Enable object tracking:**
   ```bash
   python main.py --source webcam --tracking  # Helps with false positives over time
   ```

5. **Best practice for maximum accuracy on webcam:**
   ```bash
   python main.py --source webcam --model-variant yolov8l --confidence 0.7 --iou 0.55 --tracking
   ```

6. **For comparing different models quickly:**
   ```bash
   # Quick test with nano (fastest)
   python main.py --source webcam --model-variant yolov8n --no-display
   
   # Test with medium (high accuracy)
   python main.py --source webcam --model-variant yolov8m --no-display
   
   # Test with large (maximum accuracy)
   python main.py --source webcam --model-variant yolov8l --no-display
   ```

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

### Quick Configuration (via Command Line) ✨

The easiest way is to use command-line arguments:

```bash
# Change model
python main.py --source webcam --model-variant yolov8m

# Show all available models
python main.py --list-models

# Customize thresholds
python main.py --source webcam --model-variant yolov8l --confidence 0.7 --iou 0.55
```

### Advanced Configuration (Edit `src/config.py`)

For permanent configuration changes, edit the config file:

```python
# Model Selection (change DEFAULT_MODEL to one of the variants)
DEFAULT_MODEL = 'yolov8s'  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
MODEL_PATH = f"models/{DEFAULT_MODEL}.pt"

# Getting Recommended Thresholds
# The system automatically adjusts thresholds based on selected model:
# - yolov8n: CONFIDENCE=0.7, IOU=0.55 (speed-optimized)
# - yolov8s: CONFIDENCE=0.65, IOU=0.5 (balanced, recommended)
# - yolov8m: CONFIDENCE=0.6, IOU=0.45 (accuracy-optimized)
# - yolov8l: CONFIDENCE=0.55, IOU=0.45 (high accuracy)
# - yolov8x: CONFIDENCE=0.5, IOU=0.45 (maximum accuracy)

# Detection Thresholds - Override defaults if needed
CONFIDENCE_THRESHOLD = 0.65   # Minimum confidence for detection
IOU_THRESHOLD = 0.5           # IoU threshold for NMS

# Input Configuration
INPUT_SIZE = (640, 640)       # Model input size

# Webcam Settings
WEBCAM_SOURCE = 0             # Device ID (0 = default webcam)
WEBCAM_WIDTH = 1280
WEBCAM_HEIGHT = 720

# Alert Settings
ALERT_ON_PERSON_DETECTION = False
PERSON_COUNT_THRESHOLD = 5

# Class Colors (BGR format)
CLASS_COLORS = {
    'person': (0, 255, 0),      # Green
    'vehicle': (255, 0, 0),     # Blue
    'animal': (0, 255, 255),    # Yellow
    'object': (0, 0, 255),      # Red
}
```

### Performance Tuning Tips

**For Maximum Accuracy:**
1. Use larger models: `yolov8l` or `yolov8x`
2. Decrease confidence threshold: 0.5-0.6
3. Increase IOU threshold: 0.5-0.6
4. Enable tracking for consistency across frames
5. Example: `python main.py --source webcam --model-variant yolov8x --confidence 0.5 --tracking`

**For Maximum Speed:**
1. Use smaller models: `yolov8n` or `yolov8s`
2. Increase confidence threshold: 0.7-0.8
3. Decrease IOU threshold: 0.4-0.45
4. Disable tracking
5. Run on GPU if available
6. Example: `python main.py --source webcam --model-variant yolov8n --confidence 0.8 --no-display`

**For Balanced Performance:**
1. Use `yolov8s` (default, recommended)
2. Confidence: 0.65, IOU: 0.5
3. Enable tracking for better stability
4. Example: `python main.py --source webcam --model-variant yolov8s --tracking`

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
