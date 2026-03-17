"""
Vision Agent - Combines YOLOv8 for fast object detection
with Gemini Vision for deep scene understanding
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
import google.generativeai as genai
from PIL import Image

from config import (
    YOLO_MODEL,
    GEMINI_API_KEY,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)


class VisionAgent:
    """
    Vision Agent responsible for:
    - Fast object detection using YOLOv8
    - Deep scene understanding using Gemini Vision
    - Spatial awareness and object relationships
    - Danger detection (fire, smoke, obstacles)
    - Emotion detection
    """

    def __init__(self):
        # Initialize YOLOv8 for fast object detection
        self.yolo_model = YOLO(YOLO_MODEL)
        
        # Initialize Gemini Vision for deep understanding
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.vision_model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.vision_model = None
            
        # Class names from COCO dataset (YOLOv8 default)
        self.class_names = self.yolo_model.names
        
        # Store recent detections for scene change detection
        self.recent_detections = []
        self.max_recent = 5

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Fast object detection using YOLOv8
        
        Args:
            frame: Input image frame (BGR format from OpenCV)
            
        Returns:
            List of detected objects with bounding boxes and confidence
        """
        # Run YOLOv8 inference
        results = self.yolo_model(frame, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Calculate center position
                    center_x = (x1 + x2) / 2
                    position = self._get_spatial_position(center_x, frame.shape[1])
                    
                    detections.append({
                        'class': self.class_names[class_id],
                        'class_id': class_id,
                        'confidence': confidence,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'position': position,  # left, center, right
                        'center': (float(center_x), float((y1 + y2) / 2))
                    })
        
        # Update recent detections for scene change tracking
        self.recent_detections.append(detections)
        if len(self.recent_detections) > self.max_recent:
            self.recent_detections.pop(0)
        
        return detections

    def analyze_scene(self, frame: np.ndarray, 
                     detections: Optional[List[Dict]] = None) -> str:
        """
        Deep scene understanding using Gemini Vision
        
        Args:
            frame: Input image frame
            detections: Optional YOLOv8 detections to provide context
            
        Returns:
            Natural language description of the scene
        """
        if not self.vision_model:
            return "Gemini API key not configured. Cannot perform scene analysis."
        
        # Convert BGR to RGB for Gemini
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Build prompt with optional YOLO detections as context
        prompt = self._build_scene_analysis_prompt(detections)
        
        try:
            response = self.vision_model.generate_content([prompt, pil_image])
            return response.text.strip()
        except Exception as e:
            return f"Error in scene analysis: {str(e)}"

    def detect_dangers(self, frame: np.ndarray, 
                      detections: List[Dict]) -> List[str]:
        """
        Detect dangerous situations
        
        Args:
            frame: Input image frame
            detections: YOLOv8 detections
            
        Returns:
            List of danger warnings
        """
        warnings = []
        
        # Check for fire/smoke using Gemini Vision
        if self.vision_model:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            danger_prompt = """Analyze this image for potential dangers:
            - Fire or smoke
            - People too close to camera
            - Obstacles in pathway
            - Any hazardous situations
            
            If you detect any dangers, list them clearly. If no dangers, say 'No dangers detected'."""
            
            try:
                response = self.vision_model.generate_content([danger_prompt, pil_image])
                if "no dangers" not in response.text.lower():
                    warnings.append(response.text.strip())
            except Exception:
                pass
        
        # Check for people very close (potential collision)
        for det in detections:
            if det['class'] == 'person':
                bbox = det['bbox']
                person_height = bbox[3] - bbox[1]
                if person_height > FRAME_HEIGHT * 0.7:  # Person is very close
                    warnings.append("Warning: Person detected very close to you!")
        
        return warnings

    def detect_emotions(self, frame: np.ndarray) -> Optional[str]:
        """
        Detect emotions of people in the frame
        
        Args:
            frame: Input image frame
            
        Returns:
            Emotional state description or None
        """
        if not self.vision_model:
            return None
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        emotion_prompt = """Analyze the emotional state of people in this image.
        Look for signs of:
        - Stress or anxiety
        - Fatigue or tiredness
        - Happiness or contentment
        - Sadness or distress
        
        Provide a brief, empathetic observation."""
        
        try:
            response = self.vision_model.generate_content([emotion_prompt, pil_image])
            return response.text.strip()
        except Exception:
            return None

    def detect_scene_changes(self) -> List[str]:
        """
        Detect changes in the scene compared to recent frames
        
        Returns:
            List of detected changes
        """
        if len(self.recent_detections) < 2:
            return []
        
        changes = []
        current = set(d['class'] for d in self.recent_detections[-1])
        previous = set(d['class'] for d in self.recent_detections[-2])
        
        # New objects appeared
        new_objects = current - previous
        for obj in new_objects:
            changes.append(f"New object detected: {obj}")
        
        # Objects disappeared
        disappeared = previous - current
        for obj in disappeared:
            changes.append(f"Object no longer visible: {obj}")
        
        return changes

    def _get_spatial_position(self, center_x: float, frame_width: int) -> str:
        """Determine if object is on left, center, or right"""
        third = frame_width / 3
        if center_x < third:
            return "left"
        elif center_x > 2 * third:
            return "right"
        else:
            return "center"

    def _build_scene_analysis_prompt(self, detections: Optional[List[Dict]]) -> str:
        """Build prompt for Gemini Vision with optional YOLO context"""
        base_prompt = """Describe this scene in detail. Include:
        - What objects are present and their spatial relationships
        - What activities are happening
        - The overall context and setting
        - Any notable details
        
        Be concise but informative. This will be spoken to a visually impaired user."""
        
        if detections:
            yolo_context = f"\n\nI've already detected these objects: {', '.join(d['class'] for d in detections)}"
            yolo_context += "\nUse this as a starting point but provide deeper context and meaning."
            return base_prompt + yolo_context
        
        return base_prompt

    def get_object_locations(self, detections: List[Dict]) -> str:
        """
        Generate natural language description of object locations
        
        Args:
            detections: List of detected objects
            
        Returns:
            Natural language description of spatial layout
        """
        if not detections:
            return "No objects detected in front of you."
        
        descriptions = []
        for det in detections[:5]:  # Limit to top 5
            position = det['position']
            obj_class = det['class']
            confidence = det['confidence']
            
            if confidence > 0.8:
                descriptions.append(f"{obj_class} on your {position}")
            else:
                descriptions.append(f"possibly {obj_class} on your {position}")
        
        return "I can see: " + ", ".join(descriptions) + "."
