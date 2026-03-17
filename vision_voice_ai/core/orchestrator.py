"""
Master Orchestrator - Routes user intents to appropriate agents
using Groq + Llama3 for fast intent detection
"""
import json
from typing import Dict, Optional, Tuple
from groq import Groq

from config import GROQ_API_KEY, INTENT_CONFIDENCE_THRESHOLD
from agents.vision_agent import VisionAgent
from agents.memory_agent import MemoryAgent
from agents.chat_agent import ChatAgent
from agents.web_agent import WebAgent


class MasterOrchestrator:
    """
    Master Orchestrator responsible for:
    - Fast intent detection using Groq + Llama3
    - Routing requests to appropriate agents
    - Coordinating multi-agent responses
    - Managing agent interactions
    """

    def __init__(self):
        # Initialize Groq client for intent detection
        if GROQ_API_KEY:
            self.client = Groq(api_key=GROQ_API_KEY)
            self.model = "llama3-70b-8192"
        else:
            self.client = None
        
        # Initialize all agents
        self.vision_agent = VisionAgent()
        self.memory_agent = MemoryAgent()
        self.chat_agent = ChatAgent()
        self.web_agent = WebAgent()
        
        # Intent categories
        self.intent_categories = [
            "VISION_QUERY",      # Questions about what's in front of user
            "OBJECT_LOCATION",   # Where is X?
            "SCENE_DESCRIPTION", # Describe the scene
            "DANGER_CHECK",      # Is it safe?
            "GENERAL_CHAT",      # General conversation
            "FACT_QUESTION",     # Factual questions
            "WEB_SEARCH",        # Needs internet search
            "MEMORY_QUERY",      # Questions about past
            "EMOTION_CHECK",     # How do I look?
            "UNKNOWN"            # Unclear intent
        ]
        
        # System prompt for intent classification
        self.intent_system_prompt = f"""You are an intent classifier for a voice AI assistant.
Classify user queries into one of these categories:
{chr(10).join(self.intent_categories)}

Respond ONLY with a JSON object in this format:
{{
    "intent": "CATEGORY_NAME",
    "confidence": 0.0-1.0,
    "requires_vision": true/false,
    "requires_web": true/false,
    "requires_memory": true/false
}}

Be strict with confidence scores. Only high confidence (>0.7) for clear intents."""

    def classify_intent(self, user_query: str) -> Dict:
        """
        Classify user intent using Groq + Llama3
        
        Args:
            user_query: User's input text
            
        Returns:
            Intent classification result
        """
        if not self.client:
            return self._rule_based_classification(user_query)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.intent_system_prompt},
                    {"role": "user", "content": user_query}
                ],
                max_tokens=100,
                temperature=0.1,  # Low temp for consistent classification
                stream=False
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                # Extract JSON from response (might have markdown formatting)
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0]
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0]
                
                intent_result = json.loads(result_text)
                
                # Validate required fields
                required_fields = ["intent", "confidence", "requires_vision"]
                for field in required_fields:
                    if field not in intent_result:
                        intent_result[field] = "UNKNOWN" if field == "intent" else 0.5
                
                return intent_result
                
            except json.JSONDecodeError:
                # Fallback to rule-based if JSON parsing fails
                return self._rule_based_classification(user_query)
                
        except Exception as e:
            print(f"Intent classification error: {e}")
            return self._rule_based_classification(user_query)

    def _rule_based_classification(self, user_query: str) -> Dict:
        """Fallback rule-based intent classification"""
        query_lower = user_query.lower()
        
        # Vision-related keywords
        vision_keywords = ["see", "look", "view", "in front", "around me", 
                          "what is", "what's", "describe", "show", "camera"]
        
        # Location keywords
        location_keywords = ["where", "location", "position", "left", "right", 
                            "center", "near", "far"]
        
        # Danger keywords
        danger_keywords = ["safe", "danger", "warning", "obstacle", "fire", 
                          "smoke", "hazard", "careful"]
        
        # Memory keywords
        memory_keywords = ["before", "earlier", "previous", "remember", 
                          "same", "again", "still"]
        
        # Web search indicators
        web_keywords = ["news", "latest", "current", "weather", "who is", 
                       "when did", "define", "meaning"]
        
        # Check for vision queries
        requires_vision = any(kw in query_lower for kw in vision_keywords)
        
        # Determine intent category
        if any(kw in query_lower for kw in danger_keywords):
            intent = "DANGER_CHECK"
            confidence = 0.9
        elif any(kw in query_lower for kw in location_keywords):
            intent = "OBJECT_LOCATION"
            confidence = 0.85
        elif any(kw in query_lower for kw in memory_keywords):
            intent = "MEMORY_QUERY"
            confidence = 0.8
        elif any(kw in query_lower for kw in web_keywords):
            intent = "WEB_SEARCH"
            confidence = 0.75
        elif requires_vision:
            if "describe" in query_lower or "scene" in query_lower:
                intent = "SCENE_DESCRIPTION"
            else:
                intent = "VISION_QUERY"
            confidence = 0.85
        elif "?" in query_lower:
            intent = "FACT_QUESTION"
            confidence = 0.7
        else:
            intent = "GENERAL_CHAT"
            confidence = 0.6
        
        return {
            "intent": intent,
            "confidence": confidence,
            "requires_vision": requires_vision,
            "requires_web": any(kw in query_lower for kw in web_keywords),
            "requires_memory": any(kw in query_lower for kw in memory_keywords)
        }

    def process_query(self, user_query: str, 
                     frame=None) -> Tuple[str, Dict]:
        """
        Process user query through appropriate agents
        
        Args:
            user_query: User's input text
            frame: Optional camera frame for vision queries
            
        Returns:
            Tuple of (response_text, metadata_dict)
        """
        metadata = {
            "timestamp": None,
            "intent": None,
            "agents_used": [],
            "processing_time": 0
        }
        
        from datetime import datetime
        start_time = datetime.now()
        
        # Step 1: Classify intent
        intent_result = self.classify_intent(user_query)
        intent = intent_result.get("intent", "UNKNOWN")
        confidence = intent_result.get("confidence", 0)
        
        metadata["intent"] = intent
        metadata["agents_used"].append("orchestrator")
        
        # Step 2: Store in memory
        self.memory_agent.add_user_message(user_query)
        
        # Step 3: Route to appropriate agent(s)
        response = ""
        
        if intent == "VISION_QUERY" and frame is not None:
            response = self._handle_vision_query(user_query, frame, metadata)
        
        elif intent == "OBJECT_LOCATION" and frame is not None:
            response = self._handle_location_query(user_query, frame, metadata)
        
        elif intent == "SCENE_DESCRIPTION" and frame is not None:
            response = self._handle_scene_description(frame, metadata)
        
        elif intent == "DANGER_CHECK" and frame is not None:
            response = self._handle_danger_check(frame, metadata)
        
        elif intent == "MEMORY_QUERY":
            response = self._handle_memory_query(user_query, metadata)
        
        elif intent == "WEB_SEARCH" or intent == "FACT_QUESTION":
            response = self._handle_web_query(user_query, intent_result, metadata)
        
        elif intent == "EMOTION_CHECK" and frame is not None:
            response = self._handle_emotion_check(frame, metadata)
        
        elif intent == "GENERAL_CHAT":
            response = self.chat_agent.chat(user_query)
            metadata["agents_used"].append("chat_agent")
        
        else:
            # Fallback to chat agent
            response = self.chat_agent.chat(user_query)
            metadata["agents_used"].append("chat_agent")
        
        # Step 4: Store response in memory
        self.memory_agent.add_ai_message(response)
        
        # Calculate processing time
        end_time = datetime.now()
        metadata["processing_time"] = (end_time - start_time).total_seconds()
        metadata["timestamp"] = end_time.isoformat()
        
        return response, metadata

    def _handle_vision_query(self, query: str, frame, metadata: Dict) -> str:
        """Handle queries about what objects are present"""
        # Detect objects
        detections = self.vision_agent.detect_objects(frame)
        
        # Store in memory
        self.memory_agent.remember_objects(detections)
        
        # Get object locations
        locations = self.vision_agent.get_object_locations(detections)
        
        metadata["agents_used"].extend(["vision_agent", "memory_agent"])
        metadata["detections"] = detections
        
        return locations

    def _handle_location_query(self, query: str, frame, metadata: Dict) -> str:
        """Handle queries about object positions"""
        detections = self.vision_agent.detect_objects(frame)
        self.memory_agent.remember_objects(detections)
        
        # Extract object name from query
        object_name = self._extract_object_name(query)
        
        if object_name:
            # Find specific object
            for det in detections:
                if object_name.lower() in det['class'].lower():
                    position = det['position']
                    return f"The {object_name} is on your {position}."
            
            return f"I don't see a {object_name} right now."
        
        return self.vision_agent.get_object_locations(detections)

    def _handle_scene_description(self, frame, metadata: Dict) -> str:
        """Handle requests to describe the scene"""
        # Fast detection first
        detections = self.vision_agent.detect_objects(frame)
        self.memory_agent.remember_objects(detections)
        
        # Deep analysis with Gemini Vision
        scene_analysis = self.vision_agent.analyze_scene(frame, detections)
        
        # Summarize for speech
        summary = self.chat_agent.summarize_scene(scene_analysis)
        
        metadata["agents_used"].extend(["vision_agent", "memory_agent", "chat_agent"])
        metadata["scene_analysis"] = scene_analysis
        
        return summary

    def _handle_danger_check(self, frame, metadata: Dict) -> str:
        """Check for dangers in the environment"""
        detections = self.vision_agent.detect_objects(frame)
        warnings = self.vision_agent.detect_dangers(frame, detections)
        
        metadata["agents_used"].append("vision_agent")
        metadata["warnings"] = warnings
        
        if warnings:
            return "⚠️ Warning! " + " ".join(warnings)
        else:
            return "No dangers detected. Your surroundings appear safe."

    def _handle_memory_query(self, query: str, metadata: Dict) -> str:
        """Handle queries about past conversations or objects"""
        response = self.memory_agent.query_memory(query)
        metadata["agents_used"].append("memory_agent")
        return response

    def _handle_web_query(self, query: str, intent_result: Dict, 
                         metadata: Dict) -> str:
        """Handle factual questions requiring web search"""
        response = self.web_agent.get_answer(query)
        metadata["agents_used"].append("web_agent")
        return response

    def _handle_emotion_check(self, frame, metadata: Dict) -> str:
        """Detect and report emotions"""
        emotion = self.vision_agent.detect_emotions(frame)
        metadata["agents_used"].append("vision_agent")
        
        if emotion:
            return emotion
        else:
            return "I can't clearly detect emotions right now."

    def _extract_object_name(self, query: str) -> str:
        """Extract object name from location query"""
        # Simple extraction - could be enhanced with NLP
        keywords = ["where", "is", "the", "a", "an", "my", "your"]
        words = query.lower().split()
        
        # Remove question words
        filtered = [w for w in words if w not in keywords]
        
        if filtered:
            return " ".join(filtered).replace("?", "").strip()
        
        return ""

    def get_all_detections(self, frame) -> Dict:
        """Get comprehensive detection results"""
        detections = self.vision_agent.detect_objects(frame)
        scene_changes = self.vision_agent.detect_scene_changes()
        
        return {
            "objects": detections,
            "scene_changes": scene_changes,
            "recent_objects": self.memory_agent.get_recent_objects()
        }

    def proactive_notification(self, frame) -> Optional[str]:
        """Generate proactive notifications about scene changes"""
        changes = self.vision_agent.detect_scene_changes()
        
        if changes:
            return "Heads up: " + " ".join(changes[:2])
        
        return None

    def reset_all_agents(self):
        """Reset all agent states"""
        self.memory_agent.reset_memory()
        self.chat_agent.reset_conversation()
