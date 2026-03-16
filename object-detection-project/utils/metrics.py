"""
Performance metrics module for the Object Detection System
Calculates and tracks detection performance metrics
"""

import time
from typing import List, Dict, Optional
from collections import deque
import numpy as np


class FPSCounter:
    """
    Frames Per Second counter with smoothing.
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter.
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = None
    
    def tick(self) -> float:
        """
        Record a frame timestamp and return current FPS.
        
        Returns:
            Current FPS value
        """
        current_time = time.time()
        
        if self.last_time is not None:
            elapsed = current_time - self.last_time
            if elapsed > 0:
                self.frame_times.append(1.0 / elapsed)
        
        self.last_time = current_time
        
        return self.get_fps()
    
    def get_fps(self) -> float:
        """
        Get current average FPS.
        
        Returns:
            Average FPS over the window
        """
        if not self.frame_times:
            return 0.0
        
        return sum(self.frame_times) / len(self.frame_times)
    
    def reset(self):
        """Reset FPS counter."""
        self.frame_times.clear()
        self.last_time = None


class InferenceTimer:
    """
    Timer for measuring inference latency.
    """
    
    def __init__(self):
        """Initialize inference timer."""
        self.start_time = None
        self.inference_times = deque(maxlen=100)
    
    def start(self):
        """Start timing."""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """
        Stop timing and record the duration.
        
        Returns:
            Inference time in milliseconds
        """
        if self.start_time is None:
            return 0.0
        
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        self.inference_times.append(elapsed_ms)
        self.start_time = None
        
        return elapsed_ms
    
    def get_avg_inference_time(self) -> float:
        """
        Get average inference time.
        
        Returns:
            Average inference time in milliseconds
        """
        if not self.inference_times:
            return 0.0
        
        return sum(self.inference_times) / len(self.inference_times)
    
    def reset(self):
        """Reset timer."""
        self.inference_times.clear()
        self.start_time = None


class DetectionMetrics:
    """
    Comprehensive detection metrics calculator.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.fps_counter = FPSCounter()
        self.inference_timer = InferenceTimer()
        
        # Detection statistics
        self.total_frames = 0
        self.total_detections = 0
        self.detection_counts_by_class = {}
        
        # Confidence tracking
        self.confidence_history = deque(maxlen=1000)
        
        # Per-frame metrics
        self.frame_metrics = []
    
    def record_frame(
        self,
        detections: List[Dict],
        inference_time_ms: Optional[float] = None,
    ) -> Dict:
        """
        Record metrics for a single frame.
        
        Args:
            detections: List of detections for this frame
            inference_time_ms: Inference time in milliseconds
        
        Returns:
            Frame metrics dictionary
        """
        self.total_frames += 1
        self.total_detections += len(detections)
        
        # Update class counts
        for det in detections:
            class_name = det['class_name']
            self.detection_counts_by_class[class_name] = \
                self.detection_counts_by_class.get(class_name, 0) + 1
            
            # Track confidence
            self.confidence_history.append(det['confidence'])
        
        # Calculate frame metrics
        fps = self.fps_counter.tick()
        
        if inference_time_ms is None:
            inference_time_ms = self.inference_timer.get_avg_inference_time()
        
        frame_metrics = {
            'frame_number': self.total_frames,
            'detection_count': len(detections),
            'fps': fps,
            'inference_time_ms': inference_time_ms,
            'avg_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0.0,
        }
        
        self.frame_metrics.append(frame_metrics)
        
        return frame_metrics
    
    def start_inference(self):
        """Start inference timing."""
        self.inference_timer.start()
    
    def stop_inference(self) -> float:
        """
        Stop inference timing.
        
        Returns:
            Inference time in milliseconds
        """
        return self.inference_timer.stop()
    
    def get_summary(self) -> Dict:
        """
        Get overall metrics summary.
        
        Returns:
            Summary dictionary
        """
        if self.total_frames == 0:
            return {
                'total_frames': 0,
                'total_detections': 0,
                'avg_fps': 0.0,
                'avg_inference_time_ms': 0.0,
                'avg_confidence': 0.0,
                'detections_per_frame': 0.0,
                'class_distribution': {},
            }
        
        # Calculate averages
        avg_fps = self.fps_counter.get_fps()
        avg_inference_time = self.inference_timer.get_avg_inference_time()
        avg_confidence = (
            np.mean(list(self.confidence_history)) 
            if self.confidence_history else 0.0
        )
        
        return {
            'total_frames': self.total_frames,
            'total_detections': self.total_detections,
            'avg_fps': round(avg_fps, 2),
            'avg_inference_time_ms': round(avg_inference_time, 2),
            'avg_confidence': round(avg_confidence, 4),
            'detections_per_frame': round(self.total_detections / self.total_frames, 2),
            'class_distribution': dict(self.detection_counts_by_class),
        }
    
    def reset(self):
        """Reset all metrics."""
        self.fps_counter.reset()
        self.inference_timer.reset()
        self.total_frames = 0
        self.total_detections = 0
        self.detection_counts_by_class = {}
        self.confidence_history.clear()
        self.frame_metrics.clear()


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.
    
    Args:
        bbox1: First box [x1, y1, x2, y2]
        bbox2: Second box [x1, y1, x2, y2]
    
    Returns:
        IoU value between 0 and 1
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def evaluate_detections(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict:
    """
    Evaluate detection results against ground truth.
    
    Args:
        predictions: List of predicted detections
        ground_truth: List of ground truth annotations
        iou_threshold: IoU threshold for matching
    
    Returns:
        Evaluation metrics dictionary
    """
    if not predictions and not ground_truth:
        return {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
        }
    
    matched_gt = set()
    true_positives = 0
    false_positives = 0
    
    for pred in predictions:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            
            # Only match same class
            if pred['class_name'] != gt['class_name']:
                continue
            
            iou = calculate_iou(pred['bbox'], gt['bbox'])
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            true_positives += 1
            matched_gt.add(best_gt_idx)
        else:
            false_positives += 1
    
    false_negatives = len(ground_truth) - len(matched_gt)
    
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1_score = 2 * precision * recall / max(precision + recall, 1e-6)
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1_score, 4),
        'matched_gt_count': len(matched_gt),
    }
