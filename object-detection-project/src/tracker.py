"""
Object tracking module for the Object Detection System
Tracks objects across video frames using ByteTrack algorithm
"""

import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

try:
    from ultralytics import YOLO
    # Try to import ByteTrack if available
    try:
        from boxmot import ByteTrack
        BYTETRACK_AVAILABLE = True
    except ImportError:
        BYTETRACK_AVAILABLE = False
except ImportError:
    YOLO = None
    BYTETRACK_AVAILABLE = False


class SimpleTracker:
    """
    Simple object tracker using IoU-based matching.
    Used when ByteTrack is not available.
    """
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        """
        Initialize simple tracker.
        
        Args:
            iou_threshold: IoU threshold for matching
            max_age: Maximum frames to track lost objects
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = {}  # track_id -> track_data
        self.next_track_id = 1
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Intersection over Union between two boxes.
        
        Args:
            bbox1: First box [x1, y1, x2, y2]
            bbox2: Second box [x1, y1, x2, y2]
        
        Returns:
            IoU value
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
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries
        
        Returns:
            List of tracked detections with track IDs
        """
        if not detections:
            # No detections, increment age for all tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['age'] += 1
                self.tracks[track_id]['lost'] = True
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
            return []
        
        tracked_detections = []
        used_detections = set()
        
        # Match existing tracks with detections
        for track_id, track_data in list(self.tracks.items()):
            if track_data['lost']:
                continue
            
            best_iou = 0
            best_det_idx = -1
            
            for det_idx, det in enumerate(detections):
                if det_idx in used_detections:
                    continue
                
                iou = self._calculate_iou(track_data['bbox'], det['bbox'])
                
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_det_idx >= 0:
                # Match found
                det = detections[best_det_idx]
                used_detections.add(best_det_idx)
                
                # Update track
                track_data['bbox'] = det['bbox']
                track_data['class_name'] = det['class_name']
                track_data['confidence'] = det['confidence']
                track_data['age'] = 0
                track_data['lost'] = False
                
                # Add track ID to detection
                det_with_id = det.copy()
                det_with_id['track_id'] = track_id
                tracked_detections.append(det_with_id)
            else:
                # No match, mark as lost
                track_data['lost'] = True
                track_data['age'] += 1
                if track_data['age'] > self.max_age:
                    del self.tracks[track_id]
        
        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx in used_detections:
                continue
            
            track_id = self.next_track_id
            self.next_track_id += 1
            
            self.tracks[track_id] = {
                'bbox': det['bbox'],
                'class_name': det['class_name'],
                'confidence': det['confidence'],
                'age': 0,
                'lost': False,
            }
            
            det_with_id = det.copy()
            det_with_id['track_id'] = track_id
            tracked_detections.append(det_with_id)
        
        return tracked_detections
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = {}
        self.next_track_id = 1


class ByteTrackWrapper:
    """
    Wrapper for ByteTrack algorithm.
    """
    
    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30):
        """
        Initialize ByteTrack wrapper.
        
        Args:
            track_thresh: Detection confidence threshold
            track_buffer: Frames to keep track of lost objects
        """
        if not BYTETRACK_AVAILABLE:
            raise ImportError("ByteTrack not available. Install with: pip install boxmot")
        
        self.tracker = ByteTrack(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
        )
    
    def update(self, detections: List[Dict], frame_shape: tuple) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries
            frame_shape: Frame shape (height, width)
        
        Returns:
            List of tracked detections with track IDs
        """
        if not detections:
            return []
        
        # Convert detections to format expected by ByteTrack
        det_array = []
        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            det_array.append([*bbox, conf])
        
        det_array = np.array(det_array)
        
        # Run ByteTrack update
        tracked_boxes = self.tracker.update(det_array, frame_shape)
        
        # Map back to original detections
        tracked_detections = []
        for i, tb in enumerate(tracked_boxes):
            # Find closest detection
            best_match = None
            best_iou = 0
            
            for det in detections:
                iou = self._calculate_iou(tb[:4].tolist(), det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = det
            
            if best_match:
                tracked_det = best_match.copy()
                tracked_det['track_id'] = int(tb[4])
                tracked_detections.append(tracked_det)
        
        return tracked_detections
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two boxes."""
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


def create_tracker(use_bytetrack: bool = False, **kwargs) -> SimpleTracker | ByteTrackWrapper:
    """
    Factory function to create a tracker.
    
    Args:
        use_bytetrack: Whether to use ByteTrack (if available)
        **kwargs: Additional tracker parameters
    
    Returns:
        Tracker instance
    """
    if use_bytetrack and BYTETRACK_AVAILABLE:
        return ByteTrackWrapper(**kwargs)
    else:
        return SimpleTracker(**kwargs)


def draw_tracking_info(
    frame,
    detections: List[Dict],
    show_track_id: bool = True,
) -> any:
    """
    Draw tracking information on frame.
    
    Args:
        frame: Input frame
        detections: List of tracked detections
        show_track_id: Whether to show track IDs
    
    Returns:
        Annotated frame
    """
    import cv2
    
    for det in detections:
        if 'track_id' not in det:
            continue
        
        bbox = det['bbox']
        track_id = det['track_id']
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw track ID
        if show_track_id:
            label = f"ID: {track_id}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
    
    return frame
