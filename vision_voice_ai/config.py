"""
Configuration settings for Vision Voice AI Agent
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Wake Word Settings
WAKE_WORD = os.getenv("WAKE_WORD", "Hey Vision")
WAKE_WORD_SENSITIVITY = 0.5  # 0.0 to 1.0

# Language Settings
DEFAULT_LANGUAGE = os.getenv("LANGUAGE", "en")
SUPPORTED_LANGUAGES = ["en", "hi", "es", "fr"]

# Speech Settings
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large
TTS_SPEAKER = "default"
TTS_SPEED = 1.0

# Vision Settings
YOLO_MODEL = "yolov8n.pt"  # nano model for speed
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Memory Settings
MEMORY_BUFFER_SIZE = 10  # Number of conversations to remember
MEMORY_EXPIRY_MINUTES = 60

# Agent Settings
INTENT_CONFIDENCE_THRESHOLD = 0.7
DANGER_DETECTION_ENABLED = True
EMOTION_DETECTION_ENABLED = True
SCENE_CHANGE_DETECTION_ENABLED = True

# Web Search Settings
MAX_SEARCH_RESULTS = 3

# Response Settings
MAX_RESPONSE_LENGTH = 500
RESPONSE_TIMEOUT_SECONDS = 10

# Dashboard Settings
DASHBOARD_HOST = "localhost"
DASHBOARD_PORT = 8501
DASHBOARD_THEME = "dark"

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "vision_voice_ai.log"
