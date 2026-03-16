"""
Preprocessing module for the Object Detection System
Handles input loading, resizing, normalization, and tensor conversion
"""

import cv2
import numpy as np
from typing import Union, Tuple, Optional
from pathlib import Path

from src.config import INPUT_SIZE


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from file.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Image as numpy array (BGR format) or None if failed
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image


def load_video(video_path: str) -> cv2.VideoCapture:
    """
    Open a video file.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        VideoCapture object
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    return cap


def open_webcam(source: int = 0, width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    """
    Open webcam stream.
    
    Args:
        source: Webcam device index
        width: Desired frame width
        height: Desired frame height
    
    Returns:
        VideoCapture object
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Failed to open webcam: {source}")
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    return cap


def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int] = INPUT_SIZE) -> Tuple[np.ndarray, dict]:
    """
    Preprocess a frame for YOLOv8 model input.
    
    Args:
        frame: Input frame (BGR format)
        target_size: Target size for resizing (width, height)
    
    Returns:
        Tuple of (preprocessed_frame, metadata)
        - preprocessed_frame: Resized and normalized frame (RGB format)
        - metadata: Dictionary containing original dimensions and scaling info
    """
    # Store original dimensions
    orig_height, orig_width = frame.shape[:2]
    
    # Calculate scaling ratio (maintain aspect ratio)
    scale_w = target_size[0] / orig_width
    scale_h = target_size[1] / orig_height
    scale = min(scale_w, scale_h)
    
    # Calculate new dimensions
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    
    # Resize frame
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Create canvas with target size (letterboxing)
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    # Center the resized image on canvas
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    
    # Normalize to 0-1 range
    normalized = rgb_frame.astype(np.float32) / 255.0
    
    # Metadata for post-processing
    metadata = {
        'original_width': orig_width,
        'original_height': orig_height,
        'scale': scale,
        'x_offset': x_offset,
        'y_offset': y_offset,
        'new_width': new_width,
        'new_height': new_height,
    }
    
    return normalized, metadata


def resize保持_aspect比(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Simple resize maintaining aspect ratio (alternative method).
    
    Args:
        frame: Input frame
        target_size: Target size (width, height)
    
    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def get_input_type(source: Union[str, int]) -> str:
    """
    Determine the type of input source.
    
    Args:
        source: Input source (file path or webcam index)
    
    Returns:
        String indicating input type: 'image', 'video', or 'webcam'
    """
    if isinstance(source, int):
        return 'webcam'
    
    source_path = Path(source)
    if not source_path.exists():
        return 'unknown'
    
    # Check file extension
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    
    ext = source_path.suffix.lower()
    if ext in image_extensions:
        return 'image'
    elif ext in video_extensions:
        return 'video'
    
    return 'unknown'


class InputHandler:
    """
    Unified input handler for images, videos, and webcam.
    """
    
    def __init__(self, source: Union[str, int]):
        """
        Initialize input handler.
        
        Args:
            source: Image path, video path, or webcam index
        """
        self.source = source
        self.input_type = get_input_type(source)
        self.cap = None
        self.current_frame = None
        
    def open(self) -> bool:
        """
        Open the input source.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.input_type == 'image':
                self.current_frame = load_image(self.source)
                return True
            
            elif self.input_type == 'video':
                self.cap = load_video(self.source)
                return True
            
            elif self.input_type == 'webcam':
                from src.config import WEBCAM_WIDTH, WEBCAM_HEIGHT
                self.cap = open_webcam(self.source, WEBCAM_WIDTH, WEBCAM_HEIGHT)
                return True
            
            else:
                raise ValueError(f"Unknown input type: {self.input_type}")
                
        except Exception as e:
            print(f"Error opening input: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.input_type == 'image':
            if self.current_frame is not None:
                return True, self.current_frame.copy()
            return False, None
        
        elif self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                return True, frame
            return False, None
        
        return False, None
    
    def release(self):
        """Release resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
