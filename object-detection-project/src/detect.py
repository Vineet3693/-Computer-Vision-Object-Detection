"""
Core detection module for the Object Detection System
Handles YOLOv8 model loading, inference, and raw predictions
"""

import torch
from typing import List, Dict, Optional, Union
from pathlib import Path
import time

try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not installed. Run: pip install ultralytics")
    YOLO = None

from src.config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    INPUT_SIZE,
    COCO_CLASSES,
)
from utils.logger import log_info, log_error


class DetectionEngine:
    """
    Core detection engine using YOLOv8.
    """
    
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        device: Optional[str] = None,
    ):
        """
        Initialize the detection engine.
        
        Args:
            model_path: Path to YOLOv8 model weights
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = self._select_device(device)
        self.model = None
        self.is_loaded = False
        
        log_info(f"Initializing Detection Engine with device: {self.device}")
    
    def _select_device(self, device: Optional[str]) -> str:
        """
        Select the best available device for inference.
        
        Args:
            device: Preferred device ('cuda', 'cpu', or None for auto)
        
        Returns:
            Selected device string
        """
        if device:
            return device
        
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'  # Apple Silicon
        else:
            return 'cpu'
    
    def load_model(self, model_name: Optional[str] = None) -> bool:
        """
        Load the YOLOv8 model.
        
        Args:
            model_name: Model name to load (e.g., 'yolov8s.pt')
                       If None, uses the default model_path
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if YOLO is None:
                raise ImportError("ultralytics package not installed")
            
            model_to_load = model_name if model_name else self.model_path
            
            # Create models directory if it doesn't exist
            model_dir = Path(model_to_load).parent
            model_dir.mkdir(parents=True, exist_ok=True)
            
            log_info(f"Loading YOLOv8 model: {model_to_load}")
            self.model = YOLO(model_to_load)
            
            # Move model to selected device
            self.model.to(self.device)
            self.is_loaded = True
            
            log_info(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            log_error(f"Failed to load model", e)
            self.is_loaded = False
            return False
    
    def detect(
        self,
        frame,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        classes: Optional[List[int]] = None,
        verbose: bool = False,
    ) -> List[Dict]:
        """
        Run detection on a frame.
        
        Args:
            frame: Input frame (numpy array, BGR format)
            conf_threshold: Override confidence threshold
            iou_threshold: Override IoU threshold
            classes: List of class indices to filter (None for all classes)
            verbose: Whether to print detailed output
        
        Returns:
            List of detection dictionaries containing:
            - bbox: [x1, y1, x2, y2] bounding box coordinates
            - class_id: Class index
            - class_name: Class name
            - confidence: Confidence score
        """
        if not self.is_loaded:
            log_error("Model not loaded. Call load_model() first.")
            return []
        
        try:
            # Use provided thresholds or defaults
            conf = conf_threshold if conf_threshold is not None else self.confidence_threshold
            iou = iou_threshold if iou_threshold is not None else self.iou_threshold
            
            # Run inference
            results = self.model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                classes=classes,
                verbose=verbose,
                device=self.device,
            )
            
            # Parse results
            detections = self._parse_results(results)
            return detections
            
        except Exception as e:
            log_error(f"Detection failed", e)
            return []
    
    def _parse_results(self, results) -> List[Dict]:
        """
        Parse YOLOv8 results into standardized format.
        
        Args:
            results: YOLOv8 results object
        
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for i in range(len(boxes)):
                try:
                    # Extract bounding box
                    bbox_xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                    
                    # Extract class ID and name
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
                    
                    # Extract confidence
                    confidence = float(boxes.conf[i].cpu().numpy())
                    
                    detection = {
                        'bbox': bbox_xyxy,  # [x1, y1, x2, y2]
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                    }
                    
                    detections.append(detection)
                    
                except Exception as e:
                    log_error(f"Error parsing detection box", e)
                    continue
        
        return detections
    
    def detect_batch(
        self,
        frames: List,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[List[Dict]]:
        """
        Run detection on multiple frames (batch processing).
        
        Args:
            frames: List of input frames
            conf_threshold: Override confidence threshold
            iou_threshold: Override IoU threshold
        
        Returns:
            List of detection lists (one per frame)
        """
        all_detections = []
        
        for frame in frames:
            detections = self.detect(frame, conf_threshold, iou_threshold)
            all_detections.append(detections)
        
        return all_detections
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if not self.is_loaded:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'model_path': self.model_path,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
        }


def create_detection_engine(
    model_name: str = "yolov8s.pt",
    confidence: float = 0.5,
    iou: float = 0.45,
    device: Optional[str] = None,
) -> DetectionEngine:
    """
    Factory function to create and initialize a detection engine.
    
    Args:
        model_name: YOLOv8 model variant
        confidence: Confidence threshold
        iou: IoU threshold
        device: Device preference
    
    Returns:
        Initialized DetectionEngine instance
    """
    engine = DetectionEngine(
        model_path=model_name,
        confidence_threshold=confidence,
        iou_threshold=iou,
        device=device,
    )
    engine.load_model()
    return engine
