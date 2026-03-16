"""
Visualization module for the Object Detection System
Handles drawing bounding boxes, labels, and annotations on frames
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

from src.config import (
    CLASS_COLORS,
    VEHICLE_CLASSES,
    ANIMAL_CLASSES,
    LINE_THICKNESS,
    FONT_SCALE,
    SHOW_CONFIDENCE,
    SHOW_CLASS_LABEL,
    SHOW_BBOX,
)


def get_color_for_class(class_name: str) -> Tuple[int, int, int]:
    """
    Get color for a specific class.
    
    Args:
        class_name: Name of the detected class
    
    Returns:
        BGR color tuple
    """
    if class_name == 'person':
        return CLASS_COLORS['person']
    elif class_name in VEHICLE_CLASSES:
        return CLASS_COLORS['vehicle']
    elif class_name in ANIMAL_CLASSES:
        return CLASS_COLORS['animal']
    else:
        return CLASS_COLORS['object']


def draw_bounding_box(
    frame: np.ndarray,
    bbox: List[float],
    color: Tuple[int, int, int],
    thickness: int = LINE_THICKNESS,
) -> np.ndarray:
    """
    Draw a bounding box on a frame.
    
    Args:
        frame: Input frame
        bbox: Bounding box [x1, y1, x2, y2]
        color: BGR color tuple
        thickness: Line thickness
    
    Returns:
        Frame with bounding box drawn
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Ensure coordinates are within frame bounds
    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    
    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    return frame


def draw_label(
    frame: np.ndarray,
    bbox: List[float],
    label: str,
    color: Tuple[int, int, int],
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    font_scale: float = FONT_SCALE,
) -> np.ndarray:
    """
    Draw a label above a bounding box.
    
    Args:
        frame: Input frame
        bbox: Bounding box [x1, y1, x2, y2]
        label: Label text to display
        color: Text color (BGR)
        bg_color: Background color for label
        font_scale: Font scale factor
    
    Returns:
        Frame with label drawn
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Calculate label size
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = LINE_THICKNESS
    (label_width, label_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )
    
    # Calculate label position (above the box)
    label_y = y1 - 10
    label_x = x1
    
    # Ensure label is within frame bounds
    if label_y < label_height + baseline:
        label_y = y2 + label_height + baseline
    
    # Draw label background
    cv2.rectangle(
        frame,
        (label_x, label_y - label_height - baseline),
        (label_x + label_width, label_y + baseline),
        bg_color,
        cv2.FILLED,
    )
    
    # Draw label text
    cv2.putText(
        frame,
        label,
        (label_x, label_y - baseline),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    
    return frame


def draw_confidence(
    frame: np.ndarray,
    bbox: List[float],
    confidence: float,
    color: Tuple[int, int, int],
    font_scale: float = FONT_SCALE * 0.8,
) -> np.ndarray:
    """
    Draw confidence score on a frame.
    
    Args:
        frame: Input frame
        bbox: Bounding box [x1, y1, x2, y2]
        confidence: Confidence score (0-1)
        color: Text color (BGR)
        font_scale: Font scale factor
    
    Returns:
        Frame with confidence score drawn
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Format confidence as percentage
    conf_text = f"{confidence:.2f}"
    
    # Position at bottom right of box
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = LINE_THICKNESS - 1
    (conf_width, conf_height), baseline = cv2.getTextSize(
        conf_text, font, font_scale, thickness
    )
    
    # Draw confidence text
    cv2.putText(
        frame,
        conf_text,
        (x2 - conf_width - 5, y2 - 5),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    
    return frame


def draw_detection(
    frame: np.ndarray,
    detection: Dict,
    show_label: bool = True,
    show_confidence: bool = True,
    show_bbox: bool = True,
) -> np.ndarray:
    """
    Draw a complete detection (box + label + confidence) on a frame.
    
    Args:
        frame: Input frame
        detection: Detection dictionary
        show_label: Whether to show class label
        show_confidence: Whether to show confidence score
        show_bbox: Whether to show bounding box
    
    Returns:
        Frame with detection drawn
    """
    bbox = detection['bbox']
    class_name = detection['class_name']
    confidence = detection['confidence']
    
    # Get color for this class
    color = get_color_for_class(class_name)
    
    # Draw bounding box
    if show_bbox:
        frame = draw_bounding_box(frame, bbox, color)
    
    # Build label text
    if show_label and show_confidence:
        label = f"{class_name} {confidence:.2f}"
    elif show_label:
        label = class_name
    elif show_confidence:
        label = f"{confidence:.2f}"
    else:
        label = None
    
    # Draw label
    if label and show_label:
        frame = draw_label(frame, bbox, label, color)
    
    # Draw confidence separately if not in label
    if show_confidence and not show_label:
        frame = draw_confidence(frame, bbox, confidence, color)
    
    return frame


def draw_all_detections(
    frame: np.ndarray,
    detections: List[Dict],
    show_label: bool = SHOW_CLASS_LABEL,
    show_confidence: bool = SHOW_CONFIDENCE,
    show_bbox: bool = SHOW_BBOX,
) -> np.ndarray:
    """
    Draw all detections on a frame.
    
    Args:
        frame: Input frame
        detections: List of detection dictionaries
        show_label: Whether to show class labels
        show_confidence: Whether to show confidence scores
        show_bbox: Whether to show bounding boxes
    
    Returns:
        Frame with all detections drawn
    """
    for detection in detections:
        frame = draw_detection(
            frame,
            detection,
            show_label=show_label,
            show_confidence=show_confidence,
            show_bbox=show_bbox,
        )
    
    return frame


def draw_statistics_overlay(
    frame: np.ndarray,
    stats: Dict,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = FONT_SCALE,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    line_spacing: int = 25,
) -> np.ndarray:
    """
    Draw statistics overlay on a frame.
    
    Args:
        frame: Input frame
        stats: Statistics dictionary
        position: Top-left position for overlay
        font_scale: Font scale factor
        color: Text color (BGR)
        bg_color: Background color
        line_spacing: Spacing between lines
    
    Returns:
        Frame with statistics overlay
    """
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = LINE_THICKNESS
    
    # Build statistics lines
    lines = [
        f"Total: {stats.get('total_detections', 0)}",
        f"Persons: {stats.get('person_count', 0)}",
        f"Vehicles: {stats.get('vehicle_count', 0)}",
        f"Avg Conf: {stats.get('avg_confidence', 0):.2%}",
    ]
    
    # Add FPS if available
    if 'fps' in stats:
        lines.append(f"FPS: {stats['fps']:.1f}")
    
    # Draw each line
    for i, line in enumerate(lines):
        text_y = y + (i * line_spacing)
        
        # Calculate text width for background
        (text_width, text_height), baseline = cv2.getTextSize(
            line, font, font_scale, thickness
        )
        
        # Draw background
        cv2.rectangle(
            frame,
            (x - 5, text_y - text_height - 5),
            (x + text_width + 5, text_y + baseline + 5),
            bg_color,
            cv2.FILLED,
        )
        
        # Draw text
        cv2.putText(
            frame,
            line,
            (x, text_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    
    return frame


def draw_fps_counter(
    frame: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (10, 30),
    color: Tuple[int, int, int] = (0, 255, 0),
    font_scale: float = FONT_SCALE,
) -> np.ndarray:
    """
    Draw FPS counter on a frame.
    
    Args:
        frame: Input frame
        fps: Current FPS value
        position: Position for FPS display
        color: Text color (BGR)
        font_scale: Font scale factor
    
    Returns:
        Frame with FPS counter
    """
    fps_text = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = LINE_THICKNESS
    
    cv2.putText(
        frame,
        fps_text,
        position,
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    
    return frame


def draw_alert_banner(
    frame: np.ndarray,
    alert_message: str,
    position: Tuple[int, int] = (0, 50),
    bg_color: Tuple[int, int, int] = (0, 0, 255),  # Red
    text_color: Tuple[int, int, int] = (255, 255, 255),
    font_scale: float = FONT_SCALE * 1.5,
) -> np.ndarray:
    """
    Draw alert banner on a frame.
    
    Args:
        frame: Input frame
        alert_message: Alert message text
        position: Top-left position for banner
        bg_color: Banner background color (BGR)
        text_color: Text color (BGR)
        font_scale: Font scale factor
    
    Returns:
        Frame with alert banner
    """
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = LINE_THICKNESS + 1
    
    # Calculate text size
    (text_width, text_height), baseline = cv2.getTextSize(
        alert_message, font, font_scale, thickness
    )
    
    # Draw banner background (full width)
    height, width = frame.shape[:2]
    cv2.rectangle(
        frame,
        (0, y - text_height - 10),
        (width, y + baseline + 10),
        bg_color,
        cv2.FILLED,
    )
    
    # Draw centered text
    text_x = (width - text_width) // 2
    cv2.putText(
        frame,
        alert_message,
        (text_x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )
    
    return frame


class Visualizer:
    """
    Unified visualizer for detection results.
    """
    
    def __init__(
        self,
        show_label: bool = SHOW_CLASS_LABEL,
        show_confidence: bool = SHOW_CONFIDENCE,
        show_bbox: bool = SHOW_BBOX,
        line_thickness: int = LINE_THICKNESS,
        font_scale: float = FONT_SCALE,
    ):
        """
        Initialize visualizer.
        
        Args:
            show_label: Whether to show class labels
            show_confidence: Whether to show confidence scores
            show_bbox: Whether to show bounding boxes
            line_thickness: Thickness of drawn lines
            font_scale: Scale factor for fonts
        """
        self.show_label = show_label
        self.show_confidence = show_confidence
        self.show_bbox = show_bbox
        self.line_thickness = line_thickness
        self.font_scale = font_scale
    
    def visualize(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        stats: Optional[Dict] = None,
        fps: Optional[float] = None,
        alerts: Optional[List[Dict]] = None,
    ) -> np.ndarray:
        """
        Create complete visualization of detection results.
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            stats: Optional statistics dictionary
            fps: Optional FPS value
            alerts: Optional list of alerts
        
        Returns:
            Annotated frame
        """
        # Draw all detections
        annotated = draw_all_detections(
            frame,
            detections,
            show_label=self.show_label,
            show_confidence=self.show_confidence,
            show_bbox=self.show_bbox,
        )
        
        # Draw statistics overlay
        if stats:
            annotated = draw_statistics_overlay(annotated, stats)
        
        # Draw FPS counter
        if fps is not None:
            annotated = draw_fps_counter(annotated, fps)
        
        # Draw alert banners
        if alerts:
            for i, alert in enumerate(alerts):
                y_pos = 50 + (i * 60)
                annotated = draw_alert_banner(
                    annotated,
                    alert['message'],
                    position=(0, y_pos),
                )
        
        return annotated
