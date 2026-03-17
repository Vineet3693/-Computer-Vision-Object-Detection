"""
Streamlit Dashboard - Interactive web interface for Vision Voice AI
Features:
- Live camera feed
- Real-time object detection visualization
- Conversation history
- Voice activity status
- System controls
"""
import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    GEMINI_API_KEY,
    GROQ_API_KEY,
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    DASHBOARD_HOST,
    DASHBOARD_PORT
)
from core.orchestrator import MasterOrchestrator
from utils.speech_to_text import SpeechToText
from utils.text_to_speech import TextToSpeech


# Page configuration
st.set_page_config(
    page_title="Vision Voice AI Assistant",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .danger-alert {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = None
    if 'stt' not in st.session_state:
        st.session_state.stt = None
    if 'tts' not in st.session_state:
        st.session_state.tts = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'language' not in st.session_state:
        st.session_state.language = "en"


def load_models():
    """Load AI models and initialize agents"""
    with st.spinner("Loading AI models... This may take a minute."):
        try:
            # Initialize orchestrator (loads all agents)
            st.session_state.orchestrator = MasterOrchestrator()
            
            # Initialize speech-to-text
            st.session_state.stt = SpeechToText()
            
            # Initialize text-to-speech
            st.session_state.tts = TextToSpeech()
            
            st.success("✅ All models loaded successfully!")
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            return False


def draw_detections(frame, detections):
    """Draw bounding boxes on frame"""
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        label = f"{det['class']} ({det['confidence']:.2f})"
        position = det['position']
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw position indicator
        cv2.putText(frame, position, (x1, y2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return frame


def main():
    """Main dashboard function"""
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">👁️ Vision Voice AI Assistant</p>', 
                unsafe_allow_html=True)
    st.markdown("### Dual-LLM Multi-Agent System for Visually Impaired Assistance")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # API Key Status
        st.subheader("API Status")
        col1, col2 = st.columns(2)
        with col1:
            if GEMINI_API_KEY:
                st.success("✅ Gemini")
            else:
                st.warning("⚠️ Gemini")
        with col2:
            if GROQ_API_KEY:
                st.success("✅ Groq")
            else:
                st.warning("⚠️ Groq")
        
        # Language Selection
        st.subheader("Language")
        language = st.selectbox(
            "Select Language",
            ["English", "Hindi", "Spanish", "French"],
            index=["English", "Hindi", "Spanish", "French"].index(
                {"en": "English", "hi": "Hindi", "es": "Spanish", 
                 "fr": "French"}.get(st.session_state.language, "English")
            )
        )
        lang_map = {"English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr"}
        st.session_state.language = lang_map[language]
        
        # Model Loading
        st.subheader("Models")
        if st.session_state.orchestrator is None:
            if st.button("🚀 Load AI Models", type="primary"):
                load_models()
        else:
            st.success("✅ Models Loaded")
            if st.button("🔄 Reset System"):
                st.session_state.orchestrator.reset_all_agents()
                st.session_state.conversation_history = []
                st.rerun()
        
        # System Stats
        if st.session_state.orchestrator:
            st.subheader("📊 Session Stats")
            summary = st.session_state.orchestrator.memory_agent.get_session_summary()
            st.metric("Interactions", summary['total_interactions'])
            st.metric("Objects Remembered", summary['objects_remembered'])
            st.metric("Session Duration", summary['session_duration'].split('.')[0])
        
        # Help
        with st.expander("ℹ️ How to Use"):
            st.markdown("""
            **Voice Commands:**
            - "What do you see?" - Describe surroundings
            - "Where is the [object]?" - Locate specific object
            - "Is it safe?" - Check for dangers
            - "Describe the scene" - Detailed analysis
            - "What's new?" - Scene changes
            
            **Features:**
            - 🎯 Real-time object detection
            - 🧠 Dual LLM (Groq + Gemini)
            - 💬 Natural conversation
            - ⚠️ Danger detection
            - 🌍 Multilingual support
            """)
    
    # Main content area
    if st.session_state.orchestrator is None:
        st.info("👈 Click 'Load AI Models' in the sidebar to get started!")
        return
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        
        # Camera placeholder
        camera_placeholder = st.empty()
        
        # Start/Stop camera
        if st.button("📷 Start Camera" if not st.session_state.is_running 
                     else "⏹️ Stop Camera"):
            st.session_state.is_running = not st.session_state.is_running
            st.rerun()
        
        if st.session_state.is_running:
            # Initialize camera
            cap = cv2.VideoCapture(CAMERA_INDEX)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            
            if not cap.isOpened():
                st.error("❌ Cannot open camera")
                st.session_state.is_running = False
                st.rerun()
            
            # Processing placeholder
            status_placeholder = st.empty()
            response_placeholder = st.empty()
            
            # Continuous processing loop (limited iterations for Streamlit)
            for _ in range(10):  # Process 10 frames then stop
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get current timestamp
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # Process frame through orchestrator
                detections = st.session_state.orchestrator.vision_agent.detect_objects(frame)
                
                # Draw detections
                annotated_frame = draw_detections(frame.copy(), detections)
                
                # Convert BGR to RGB for Streamlit
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                camera_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                
                # Update status
                objects_detected = [d['class'] for d in detections[:5]]
                status_placeholder.info(
                    f"🔍 Detected: {', '.join(objects_detected) if objects_detected else 'No objects'} | "
                    f"Time: {timestamp}"
                )
                
                # Store detections in memory
                st.session_state.orchestrator.memory_agent.remember_objects(detections)
            
            cap.release()
            st.session_state.is_running = False
    
    with col2:
        st.subheader("💬 Conversation")
        
        # Display conversation history
        conversation_container = st.container()
        with conversation_container:
            for msg in st.session_state.conversation_history[-10:]:
                if msg['role'] == 'user':
                    st.chat_message("user").write(msg['content'])
                else:
                    st.chat_message("assistant").write(msg['content'])
        
        # Text input for testing
        user_input = st.chat_input("Type a message...")
        
        if user_input:
            # Add to history
            st.session_state.conversation_history.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now()
            })
            
            # Process query
            with st.spinner("Thinking..."):
                response, metadata = st.session_state.orchestrator.process_query(user_input)
            
            # Add response to history
            st.session_state.conversation_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now()
            })
            
            # Display intent detection
            with st.expander(f"🧠 Intent: {metadata.get('intent', 'UNKNOWN')}"):
                st.json(metadata)
            
            # Speak response if TTS available
            if st.session_state.tts and st.checkbox("🔊 Enable Speech Output"):
                with st.spinner("Speaking..."):
                    st.session_state.tts.speak(response, block=False)
            
            st.rerun()
        
        # Quick action buttons
        st.subheader("⚡ Quick Actions")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("👀 What do I see?", use_container_width=True):
                user_input = "What do I see in front of me?"
                st.session_state.conversation_history.append({
                    'role': 'user', 'content': user_input,
                    'timestamp': datetime.now()
                })
                with st.spinner("Analyzing..."):
                    # Need to capture frame first
                    cap = cv2.VideoCapture(CAMERA_INDEX)
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        response, _ = st.session_state.orchestrator.process_query(
                            user_input, frame=frame
                        )
                    else:
                        response = "Camera not available. Please start the camera first."
                    
                    st.session_state.conversation_history.append({
                        'role': 'assistant', 'content': response,
                        'timestamp': datetime.now()
                    })
                    st.rerun()
        
        with col_b:
            if st.button("⚠️ Is it safe?", use_container_width=True):
                user_input = "Is it safe around me?"
                st.session_state.conversation_history.append({
                    'role': 'user', 'content': user_input,
                    'timestamp': datetime.now()
                })
                with st.spinner("Checking for dangers..."):
                    cap = cv2.VideoCapture(CAMERA_INDEX)
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        response, _ = st.session_state.orchestrator.process_query(
                            user_input, frame=frame
                        )
                    else:
                        response = "Camera not available."
                    
                    st.session_state.conversation_history.append({
                        'role': 'assistant', 'content': response,
                        'timestamp': datetime.now()
                    })
                    st.rerun()
        
        # Recent objects
        st.subheader("📦 Recent Objects")
        recent_objects = st.session_state.orchestrator.memory_agent.get_recent_objects()
        if recent_objects:
            st.write(", ".join(set(recent_objects)))
        else:
            st.info("No objects detected yet")


if __name__ == "__main__":
    main()
