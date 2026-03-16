"""
Post-processing module for the Object Detection System
Handles filtering, counting, alerting, and logging detections
"""

import csv
import os
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict

from src.config import (
    VEHICLE_CLASSES,
    ANIMAL_CLASSES,
    DETECTION_LOG_FILE,
    PERSON_COUNT_THRESHOLD,
    ALERT_ON_PERSON_DETECTION,
)
from utils.logger import log_info, log_warning


def filter_detections(
    detections: List[Dict],
    min_confidence: float = 0.5,
    classes_filter: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Filter detections by confidence and class.
    
    Args:
        detections: List of detection dictionaries
        min_confidence: Minimum confidence threshold
        classes_filter: List of class names to keep (None for all)
    
    Returns:
        Filtered list of detections
    """
    filtered = []
    
    for det in detections:
        # Check confidence
        if det['confidence'] < min_confidence:
            continue
        
        # Check class filter
        if classes_filter and det['class_name'] not in classes_filter:
            continue
        
        filtered.append(det)
    
    return filtered


def count_objects(detections: List[Dict]) -> Dict[str, int]:
    """
    Count detected objects by class.
    
    Args:
        detections: List of detection dictionaries
    
    Returns:
        Dictionary mapping class names to counts
    """
    counts = defaultdict(int)
    
    for det in detections:
        class_name = det['class_name']
        counts[class_name] += 1
    
    return dict(counts)


def categorize_detections(detections: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Categorize detections into groups (person, vehicle, animal, object).
    
    Args:
        detections: List of detection dictionaries
    
    Returns:
        Dictionary with categorized detections
    """
    categories = {
        'person': [],
        'vehicle': [],
        'animal': [],
        'object': [],
    }
    
    for det in detections:
        class_name = det['class_name']
        
        if class_name == 'person':
            categories['person'].append(det)
        elif class_name in VEHICLE_CLASSES:
            categories['vehicle'].append(det)
        elif class_name in ANIMAL_CLASSES:
            categories['animal'].append(det)
        else:
            categories['object'].append(det)
    
    return categories


def calculate_statistics(detections: List[Dict]) -> Dict:
    """
    Calculate detection statistics.
    
    Args:
        detections: List of detection dictionaries
    
    Returns:
        Dictionary containing various statistics
    """
    if not detections:
        return {
            'total_detections': 0,
            'person_count': 0,
            'vehicle_count': 0,
            'animal_count': 0,
            'object_count': 0,
            'avg_confidence': 0.0,
            'max_confidence': 0.0,
            'min_confidence': 0.0,
        }
    
    categorized = categorize_detections(detections)
    confidences = [det['confidence'] for det in detections]
    
    stats = {
        'total_detections': len(detections),
        'person_count': len(categorized['person']),
        'vehicle_count': len(categorized['vehicle']),
        'animal_count': len(categorized['animal']),
        'object_count': len(categorized['object']),
        'avg_confidence': sum(confidences) / len(confidences),
        'max_confidence': max(confidences),
        'min_confidence': min(confidences),
        'class_counts': count_objects(detections),
    }
    
    return stats


def check_alerts(stats: Dict) -> List[Dict]:
    """
    Check for alert conditions based on detection statistics.
    
    Args:
        stats: Detection statistics dictionary
    
    Returns:
        List of alert dictionaries
    """
    alerts = []
    
    # Person count alert
    if ALERT_ON_PERSON_DETECTION and stats['person_count'] >= PERSON_COUNT_THRESHOLD:
        alerts.append({
            'type': 'PERSON_COUNT_THRESHOLD',
            'message': f"Person count ({stats['person_count']}) exceeded threshold ({PERSON_COUNT_THRESHOLD})",
            'severity': 'warning',
            'timestamp': datetime.now().isoformat(),
        })
    
    return alerts


def save_to_csv(
    detections: List[Dict],
    output_file: str = DETECTION_LOG_FILE,
    frame_id: Optional[int] = None,
    source_type: str = 'unknown',
):
    """
    Save detection results to CSV file.
    
    Args:
        detections: List of detection dictionaries
        output_file: Path to output CSV file
        frame_id: Frame identifier (for video/webcam)
        source_type: Type of input source
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(output_file)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(output_file, 'a', newline='') as f:
        fieldnames = [
            'timestamp', 'frame_id', 'source_type',
            'class_id', 'class_name', 'confidence',
            'x1', 'y1', 'x2', 'y2', 'width', 'height'
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header if new file
        if not file_exists:
            writer.writeheader()
        
        # Write each detection
        for det in detections:
            bbox = det['bbox']
            row = {
                'timestamp': timestamp,
                'frame_id': frame_id if frame_id is not None else '',
                'source_type': source_type,
                'class_id': det['class_id'],
                'class_name': det['class_name'],
                'confidence': f"{det['confidence']:.4f}",
                'x1': f"{bbox[0]:.2f}",
                'y1': f"{bbox[1]:.2f}",
                'x2': f"{bbox[2]:.2f}",
                'y2': f"{bbox[3]:.2f}",
                'width': f"{bbox[2] - bbox[0]:.2f}",
                'height': f"{bbox[3] - bbox[1]:.2f}",
            }
            writer.writerow(row)


def format_detection_summary(stats: Dict) -> str:
    """
    Format detection statistics as a human-readable summary.
    
    Args:
        stats: Detection statistics dictionary
    
    Returns:
        Formatted summary string
    """
    summary_lines = [
        f"=== Detection Summary ===",
        f"Total Detections: {stats['total_detections']}",
        f"Persons: {stats['person_count']}",
        f"Vehicles: {stats['vehicle_count']}",
        f"Animals: {stats['animal_count']}",
        f"Other Objects: {stats['object_count']}",
        f"Avg Confidence: {stats['avg_confidence']:.2%}",
    ]
    
    # Add class breakdown if there are detections
    if stats['total_detections'] > 0 and 'class_counts' in stats:
        summary_lines.append("\nClass Breakdown:")
        for class_name, count in sorted(stats['class_counts'].items(), key=lambda x: -x[1]):
            summary_lines.append(f"  - {class_name}: {count}")
    
    return '\n'.join(summary_lines)


class PostProcessor:
    """
    Unified post-processor for detection results.
    """
    
    def __init__(self, log_file: str = DETECTION_LOG_FILE):
        """
        Initialize post-processor.
        
        Args:
            log_file: Path to detection log CSV
        """
        self.log_file = log_file
        self.frame_count = 0
        self.total_detections = 0
        self.alert_history = []
    
    def process(
        self,
        detections: List[Dict],
        source_type: str = 'unknown',
        save_log: bool = True,
    ) -> Dict:
        """
        Process detection results.
        
        Args:
            detections: List of detection dictionaries
            source_type: Type of input source
            save_log: Whether to save to CSV
        
        Returns:
            Processing results dictionary
        """
        # Increment frame count
        self.frame_count += 1
        
        # Calculate statistics
        stats = calculate_statistics(detections)
        
        # Update total
        self.total_detections += stats['total_detections']
        
        # Check alerts
        alerts = check_alerts(stats)
        self.alert_history.extend(alerts)
        
        # Log alerts
        for alert in alerts:
            log_warning(f"ALERT: {alert['message']}")
        
        # Save to CSV
        if save_log and detections:
            save_to_csv(
                detections=detections,
                output_file=self.log_file,
                frame_id=self.frame_count,
                source_type=source_type,
            )
        
        return {
            'stats': stats,
            'alerts': alerts,
            'frame_count': self.frame_count,
            'categorized': categorize_detections(detections),
        }
    
    def reset(self):
        """Reset processor state."""
        self.frame_count = 0
        self.total_detections = 0
        self.alert_history = []
    
    def get_summary(self) -> Dict:
        """
        Get processing summary.
        
        Returns:
            Summary dictionary
        """
        return {
            'total_frames_processed': self.frame_count,
            'total_detections': self.total_detections,
            'total_alerts': len(self.alert_history),
            'avg_detections_per_frame': self.total_detections / max(self.frame_count, 1),
        }
