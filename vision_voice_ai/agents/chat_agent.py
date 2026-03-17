"""
Chat Agent - Handles general conversation using Groq + Llama3
for fast, intelligent responses
"""
from typing import Optional, List
from groq import Groq

from config import GROQ_API_KEY


class ChatAgent:
    """
    Chat Agent responsible for:
    - General conversation not related to vision
    - Fast Q&A using Groq + Llama3
    - Fallback responses when other agents can't help
    - Natural dialogue flow
    """

    def __init__(self):
        # Initialize Groq client for ultra-fast LLM inference
        if GROQ_API_KEY:
            self.client = Groq(api_key=GROQ_API_KEY)
            self.model = "llama3-70b-8192"  # Fast and capable
        else:
            self.client = None
            self.model = None
        
        # System prompt for consistent behavior
        self.system_prompt = """You are a helpful AI assistant for visually impaired users.
Your responses should be:
- Clear and concise (meant to be spoken aloud)
- Empathetic and supportive
- Informative without being overwhelming
- Focused on practical, actionable information

Avoid visual descriptions unless specifically asked.
Keep responses under 3 sentences when possible."""

        # Conversation context for better follow-ups
        self.conversation_history: List[dict] = []
        self.max_history = 5

    def chat(self, user_message: str, 
             context: Optional[str] = None) -> str:
        """
        Generate a response to user's message
        
        Args:
            user_message: User's input text
            context: Optional additional context from other agents
            
        Returns:
            AI response text
        """
        if not self.client:
            return self._fallback_response(user_message)
        
        # Build messages array
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add recent conversation history
        messages.extend(self.conversation_history[-self.max_history:])
        
        # Add context if provided
        if context:
            context_msg = f"Context: {context}"
            messages.append({"role": "system", "content": context_msg})
        
        # Add user message
        messages.append({"role": "user", "content": user_message})
        
        try:
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=150,
                temperature=0.7,
                top_p=1.0,
                stream=False
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Update conversation history
            self._update_history(user_message, ai_response)
            
            return ai_response
            
        except Exception as e:
            return f"I encountered an error: {str(e)}. Let me try again."

    def answer_question(self, question: str) -> str:
        """
        Answer a specific question
        
        Args:
            question: User's question
            
        Returns:
            Answer text
        """
        prompt = f"Please answer this question clearly and concisely: {question}"
        return self.chat(prompt)

    def explain_objects(self, objects: List[str], 
                       relationships: Optional[str] = None) -> str:
        """
        Explain detected objects in a helpful way
        
        Args:
            objects: List of detected object names
            relationships: Optional spatial relationships
            
        Returns:
            Helpful explanation
        """
        if not objects:
            return "I don't see any objects right now."
        
        context = f"Detected objects: {', '.join(objects)}"
        if relationships:
            context += f"\nSpatial relationships: {relationships}"
        
        prompt = """Based on the objects I'm seeing, help me understand my environment.
Describe what these objects typically mean in a scene and how they might be arranged.
Keep it practical and useful for navigation."""
        
        return self.chat(prompt, context=context)

    def provide_guidance(self, situation: str) -> str:
        """
        Provide guidance for a specific situation
        
        Args:
            situation: Description of current situation
            
        Returns:
            Guidance/suggestions
        """
        prompt = f"""I'm in this situation: {situation}

Provide clear, actionable guidance. Focus on safety and practical next steps.
Keep it brief and encouraging."""
        
        return self.chat(prompt)

    def clarify_intent(self, ambiguous_query: str) -> str:
        """
        Ask clarifying questions when user intent is unclear
        
        Args:
            ambiguous_query: User's unclear query
            
        Returns:
            Clarifying question
        """
        prompt = f"""The user said: "{ambiguous_query}"

This is ambiguous. Ask ONE clear, friendly clarifying question to understand what they need.
Don't try to answer yet, just ask for clarification."""
        
        return self.chat(prompt)

    def summarize_scene(self, scene_description: str) -> str:
        """
        Create a concise summary of a scene description
        
        Args:
            scene_description: Detailed scene analysis from Vision Agent
            
        Returns:
            Concise summary suitable for speech
        """
        prompt = f"""Here's a detailed scene description:

{scene_description}

Create a concise, spoken summary in 2-3 sentences.
Focus on the most important elements for navigation and awareness.
Use natural, conversational language."""
        
        return self.chat(prompt)

    def _update_history(self, user_msg: str, ai_msg: str):
        """Update conversation history"""
        self.conversation_history.append({
            "role": "user",
            "content": user_msg
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": ai_msg
        })
        
        # Trim if too long
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]

    def _fallback_response(self, message: str) -> str:
        """Generate fallback response when Groq is unavailable"""
        message_lower = message.lower()
        
        # Simple keyword-based responses
        if any(word in message_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I'm your Vision Assistant. How can I help you today?"
        
        elif any(word in message_lower for word in ["thank", "thanks"]):
            return "You're welcome! I'm here to help whenever you need."
        
        elif "?" in message_lower:
            return "I'd love to help answer that, but I need my API key configured. Please set up your Groq API key for full functionality."
        
        else:
            return "I'm listening! Please configure your Groq API key so I can provide intelligent responses."

    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []

    def get_conversation_summary(self) -> str:
        """Get a summary of recent conversation"""
        if not self.conversation_history:
            return "No conversation history."
        
        # Get last few exchanges
        recent = self.conversation_history[-4:]
        summary_parts = []
        
        for msg in recent:
            role = "User" if msg["role"] == "user" else "AI"
            preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
            summary_parts.append(f"{role}: {preview}")
        
        return "\n".join(summary_parts)
