"""
Response Generator - Uses Gemini Pro to generate natural, 
contextual responses optimized for speech
"""
from typing import Optional, Dict
import google.generativeai as genai

from config import GEMINI_API_KEY


class ResponseGenerator:
    """
    Response Generator responsible for:
    - Converting agent outputs into natural speech-friendly responses
    - Adding appropriate emotion and tone
    - Ensuring responses are concise and clear
    - Handling multilingual responses
    """

    def __init__(self, default_language: str = "en"):
        # Initialize Gemini Pro for response generation
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None
        
        self.default_language = default_language
        
        # Response guidelines for speech optimization
        self.speech_guidelines = """
Guidelines for generating responses:
1. Keep responses concise (1-3 sentences ideal for speech)
2. Use clear, simple language
3. Avoid visual references unless specifically asked
4. Be empathetic and supportive
5. Prioritize safety-critical information first
6. Use natural, conversational tone
7. Avoid jargon and technical terms
8. Round numbers (say "about 5" instead of "4.7")
9. Use active voice
10. End with helpful suggestion when appropriate
"""

    def generate_response(self, agent_output: str, 
                         context: Optional[Dict] = None,
                         language: Optional[str] = None) -> str:
        """
        Generate a natural, speech-optimized response
        
        Args:
            agent_output: Raw output from agents
            context: Additional context (intent, user state, etc.)
            language: Target language (default: English)
            
        Returns:
            Natural language response optimized for speech
        """
        if not self.model:
            return self._simple_format(agent_output)
        
        language = language or self.default_language
        
        prompt = f"""{self.speech_guidelines}

Agent output: {agent_output}

{"Context: " + str(context) if context else ""}

Convert this into a natural, spoken response in {language}.
Keep it brief, clear, and conversational."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return self._simple_format(agent_output)

    def generate_emergency_alert(self, danger_type: str, 
                                severity: str = "high") -> str:
        """
        Generate urgent emergency alert
        
        Args:
            danger_type: Type of danger detected
            severity: Danger level (low/medium/high/critical)
            
        Returns:
            Urgent alert message
        """
        urgency_words = {
            "low": "Notice",
            "medium": "Caution",
            "high": "Warning",
            "critical": "DANGER"
        }
        
        urgency = urgency_words.get(severity, "Warning")
        
        alerts = {
            "fire": f"{urgency}! Fire or smoke detected nearby!",
            "obstacle": f"{urgency}! Obstacle in your path!",
            "person_close": f"{urgency}! Person very close to you!",
            "vehicle": f"{urgency}! Vehicle approaching!",
            "stairs": f"{urgency}! Stairs or elevation change ahead!",
            "traffic": f"{urgency}! Traffic detected!"
        }
        
        base_alert = alerts.get(danger_type.lower(), 
                               f"{urgency}! {danger_type} detected!")
        
        if severity in ["high", "critical"]:
            base_alert += " Please stop and assess the situation."
        
        return base_alert

    def generate_proactive_update(self, changes: list) -> str:
        """
        Generate proactive notification about scene changes
        
        Args:
            changes: List of detected changes
            
        Returns:
            Natural update message
        """
        if not changes:
            return ""
        
        if len(changes) == 1:
            return f"Just so you know, {changes[0].lower()}."
        
        # Multiple changes
        intro = "I've noticed some changes: "
        items = ", ".join([c.lower() for c in changes[:3]])
        return intro + items + "."

    def generate_confirmation(self, action: str) -> str:
        """
        Generate confirmation message
        
        Args:
            action: Action being confirmed
            
        Returns:
            Confirmation message
        """
        confirmations = {
            "memory_saved": "Got it, I'll remember that.",
            "search_started": "Searching for that now...",
            "analyzing": "Analyzing the scene...",
            "listening": "I'm listening...",
            "processing": "Processing your request..."
        }
        
        return confirmations.get(action.lower(), "Understood.")

    def generate_error_message(self, error_type: str, 
                              user_friendly: bool = True) -> str:
        """
        Generate user-friendly error message
        
        Args:
            error_type: Type of error
            user_friendly: Whether to use simple language
            
        Returns:
            Error message
        """
        errors = {
            "vision": "I'm having trouble seeing right now. Let me try again.",
            "speech": "I didn't catch that. Could you repeat?",
            "network": "Having connection issues. Please try again.",
            "api": "Service temporarily unavailable. Please wait a moment.",
            "timeout": "That's taking longer than expected. Let's try again.",
            "unknown": "Something went wrong. Let's try that again."
        }
        
        base_error = errors.get(error_type.lower(), errors["unknown"])
        
        if user_friendly:
            base_error += " Is there anything else I can help with?"
        
        return base_error

    def _simple_format(self, text: str) -> str:
        """Simple formatting when model is unavailable"""
        # Clean up text
        text = text.strip()
        
        # Truncate if too long
        if len(text) > 200:
            text = text[:197] + "..."
        
        # Ensure proper ending
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text

    def adjust_for_language(self, text: str, language: str) -> str:
        """
        Adjust response for specific language/culture
        
        Args:
            text: Original text
            language: Target language code
            
        Returns:
            Culturally adjusted text
        """
        # For production, would use translation API
        # This is a placeholder
        if language != "en":
            return f"[{language}] {text}"
        
        return text

    def generate_follow_up(self, topic: str) -> str:
        """
        Generate helpful follow-up suggestion
        
        Args:
            topic: Current topic
            
        Returns:
            Follow-up question or suggestion
        """
        followups = {
            "navigation": "Would you like me to describe the path ahead?",
            "objects": "Would you like to know more about any specific object?",
            "scene": "Should I keep monitoring for changes?",
            "danger": "Would you like me to continue scanning for hazards?",
            "general": "Is there anything else you'd like to know?"
        }
        
        return followups.get(topic.lower(), followups["general"])
