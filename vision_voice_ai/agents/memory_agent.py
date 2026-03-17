"""
Memory Agent - Manages conversation history and object memory
using LangChain for persistent context
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    GEMINI_API_KEY,
    MEMORY_BUFFER_SIZE,
    MEMORY_EXPIRY_MINUTES,
)


class MemoryAgent:
    """
    Memory Agent responsible for:
    - Storing conversation history
    - Remembering previously detected objects
    - Maintaining session context
    - Providing contextual responses based on history
    """

    def __init__(self):
        # Initialize LangChain conversation memory
        self.memory = ConversationBufferMemory(
            max_token_limit=MEMORY_BUFFER_SIZE * 100,
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Object memory - stores recently seen objects with timestamps
        self.object_memory: Dict[str, datetime] = {}
        
        # Session metadata
        self.session_start = datetime.now()
        self.total_interactions = 0
        
        # Initialize LLM for memory queries if API key available
        if GEMINI_API_KEY:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=GEMINI_API_KEY,
                    temperature=0.3
                )
            except Exception:
                self.llm = None
        else:
            self.llm = None

    def add_user_message(self, message: str):
        """Add user message to conversation memory"""
        self.memory.chat_memory.add_user_message(message)
        self.total_interactions += 1

    def add_ai_message(self, message: str):
        """Add AI response to conversation memory"""
        self.memory.chat_memory.add_ai_message(message)

    def add_conversation_pair(self, user_msg: str, ai_msg: str):
        """Add a complete conversation turn"""
        self.add_user_message(user_msg)
        self.add_ai_message(ai_msg)

    def remember_objects(self, detections: List[Dict]):
        """
        Store detected objects in memory with timestamps
        
        Args:
            detections: List of detected objects from Vision Agent
        """
        current_time = datetime.now()
        
        for detection in detections:
            obj_class = detection['class']
            position = detection.get('position', 'unknown')
            
            # Create unique key with position context
            key = f"{obj_class}_{position}"
            self.object_memory[key] = current_time
            
            # Also store without position for general queries
            self.object_memory[obj_class] = current_time

    def get_recent_objects(self, minutes: int = 5) -> List[str]:
        """
        Get objects detected within the last N minutes
        
        Args:
            minutes: Time window in minutes
            
        Returns:
            List of object class names
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent = [
            obj_key.split('_')[0]  # Remove position suffix
            for obj_key, timestamp in self.object_memory.items()
            if timestamp > cutoff_time
        ]
        return list(set(recent))  # Remove duplicates

    def was_object_seen_before(self, obj_class: str, 
                               position: Optional[str] = None) -> bool:
        """
        Check if an object was previously detected
        
        Args:
            obj_class: Object class name
            position: Optional position (left/center/right)
            
        Returns:
            True if object was seen before
        """
        if position:
            key = f"{obj_class}_{position}"
            return key in self.object_memory
        return obj_class in self.object_memory

    def get_object_first_seen(self, obj_class: str) -> Optional[datetime]:
        """Get timestamp when object was first detected"""
        # Check with position variants
        for key, timestamp in self.object_memory.items():
            if key.startswith(f"{obj_class}_"):
                return timestamp
        
        # Check without position
        return self.object_memory.get(obj_class)

    def get_conversation_history(self) -> str:
        """Get formatted conversation history"""
        messages = self.memory.chat_memory.messages
        history = []
        
        for msg in messages[-MEMORY_BUFFER_SIZE:]:  # Limit to recent
            role = "User" if msg.type == "human" else "AI"
            history.append(f"{role}: {msg.content}")
        
        return "\n".join(history)

    def clear_short_term_memory(self):
        """Clear recent object detections but keep conversation"""
        cutoff_time = datetime.now() - timedelta(minutes=MEMORY_EXPIRY_MINUTES)
        
        to_remove = [
            key for key, timestamp in self.object_memory.items()
            if timestamp < cutoff_time
        ]
        
        for key in to_remove:
            del self.object_memory[key]

    def reset_memory(self):
        """Completely reset all memory"""
        self.memory.clear()
        self.object_memory.clear()
        self.session_start = datetime.now()
        self.total_interactions = 0

    def get_session_summary(self) -> Dict:
        """Get summary of current session"""
        return {
            "session_duration": str(datetime.now() - self.session_start),
            "total_interactions": self.total_interactions,
            "objects_remembered": len(self.object_memory),
            "conversation_turns": len(self.memory.chat_memory.messages) // 2
        }

    def query_memory(self, question: str) -> str:
        """
        Use LLM to answer questions about conversation history
        
        Args:
            question: User's question about past conversation
            
        Returns:
            AI response based on memory
        """
        if not self.llm:
            return "Memory query requires Gemini API key."
        
        history = self.get_conversation_history()
        
        if not history:
            return "I don't have any previous conversation to reference."
        
        prompt = f"""Based on this conversation history:

{history}

Answer the user's question: {question}

If the information is not in the history, say so honestly."""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error querying memory: {str(e)}"

    def get_contextual_response(self, current_objects: List[str]) -> str:
        """
        Generate context-aware response based on current and past objects
        
        Args:
            current_objects: Objects currently detected
            
        Returns:
            Contextual statement about continuity
        """
        recent_past = self.get_recent_objects(minutes=2)
        
        # Find objects that are still present
        continuing = set(current_objects) & set(recent_past)
        
        # Find new objects
        new_objects = set(current_objects) - set(recent_past)
        
        statements = []
        
        if continuing:
            statements.append(
                f"The {', '.join(continuing)} {'is' if len(continuing) == 1 else 'are'} "
                f"still {'there' if len(continuing) == 1 else 'in view'}."
            )
        
        if new_objects:
            statements.append(
                f"I {'notice' if len(new_objects) == 1 else 'see'} "
                f"a new {', '.join(new_objects)}."
            )
        
        return " ".join(statements) if statements else ""

    def cleanup_old_memories(self):
        """Remove memories older than expiry time"""
        self.clear_short_term_memory()
        
        # Optionally trim conversation memory if too long
        messages = self.memory.chat_memory.messages
        if len(messages) > MEMORY_BUFFER_SIZE * 2:
            # Keep only recent messages
            self.memory.chat_memory.messages = messages[-MEMORY_BUFFER_SIZE:]
