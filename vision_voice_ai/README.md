# Advanced Vision + Voice AI Agent with Gemini AI & Groq

A dual-LLM multi-agent AI system that combines **Groq + Llama3** for speed-critical tasks and **Google Gemini AI** for vision understanding and deep reasoning. This project is designed as an **AI Vision Assistant for Visually Impaired People**.

## 🌟 Key Features

- **Wake Word Detection** - "Hey Vision" to activate the system
- **Dual LLM Architecture** - Groq for speed, Gemini for smart vision/reasoning
- **Multi-Agent System**:
  - Vision Agent (YOLOv8 + Gemini Vision)
  - Memory Agent (LangChain)
  - Chat Agent (Groq + Llama3)
  - Web Search Agent (DuckDuckGo)
- **Danger Detection** - Fire, smoke, obstacles, proximity alerts
- **Multilingual Support** - Hindi, English, Spanish, French
- **Emotion Detection** - Detects user stress/fatigue
- **Scene Change Detection** - Proactive notifications
- **Real-time Dashboard** - Streamlit interface with live camera feed

## 🏗 Architecture

```
User Voice → Wake Word → Speech Recognition → Master Orchestrator (Groq)
                                                      ↓
                                            Agent Router
                                                      ↓
        ┌─────────────┬─────────────┬─────────────┬─────────────┐
        ↓             ↓             ↓             ↓             ↓
   Vision Agent   Memory Agent   Chat Agent   Web Agent   Danger Detection
   (Gemini+YOLO)  (LangChain)    (Groq)       (DuckDuckGo)  (Gemini Vision)
        ↓             ↓             ↓             ↓
   Scene Analysis  Conversation  Q&A Answers   Live Info
   Object Detection History
        ↓
   Response Generator (Gemini Pro)
        ↓
   Text-to-Speech (Coqui TTS) → Speaker Output
```

## 📁 Project Structure

```
vision_voice_ai/
├── agents/
│   ├── __init__.py
│   ├── vision_agent.py      # YOLOv8 + Gemini Vision integration
│   ├── memory_agent.py      # LangChain memory management
│   ├── chat_agent.py        # Groq + Llama3 conversation
│   └── web_agent.py         # DuckDuckGo search integration
├── core/
│   ├── __init__.py
│   ├── orchestrator.py      # Master intent detection & routing
│   ├── wake_word.py         # Picovoice Porcupine wake word
│   └── response_generator.py # Gemini Pro response generation
├── utils/
│   ├── __init__.py
│   ├── speech_to_text.py    # Whisper STT
│   ├── text_to_speech.py    # Coqui TTS
│   ├── danger_detection.py  # Safety alerts
│   └── emotion_detection.py # Emotion analysis
├── models/
│   └── yolo_weights/        # YOLOv8 model weights
├── dashboard/
│   ├── __init__.py
│   └── app.py               # Streamlit interface
├── config.py                # Configuration settings
├── requirements.txt         # Dependencies
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Google Gemini API key
- Groq API key
- Webcam/microphone

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
WAKE_WORD="Hey Vision"
LANGUAGE=en
```

### Run the System

```bash
# Launch the Streamlit dashboard
streamlit run dashboard/app.py
```

## 🎯 Use Cases

1. **Assistive Technology** - Help visually impaired users navigate spaces
2. **Smart Environment Monitoring** - Detect dangers and alert users
3. **Interactive Companion** - Natural conversation with context memory
4. **Educational Tool** - Describe scenes, objects, and emotions

## 🤖 Dual LLM Strategy

| Layer | Model | Purpose |
|-------|-------|---------|
| Speed Layer | Groq + Llama3 | Fast intent detection & routing |
| Vision Layer | Gemini Vision | Deep image & scene understanding |
| Reasoning Layer | Gemini Pro | Smart response generation |
| Memory Layer | LangChain | Conversation & object memory |
| Detection Layer | YOLOv8 | Real-time object detection |

## 🌍 Social Impact

This project is designed as an **AI Vision Assistant for Visually Impaired People**, providing:

- ✅ Real-time environment description
- ✅ Danger warnings (fire, obstacles, people)
- ✅ Object location and spatial awareness
- ✅ Conversational memory for context
- ✅ Multilingual support for accessibility

## 📊 Technical Stack

- **LLMs**: Groq (Llama3), Google Gemini (Vision + Pro)
- **Object Detection**: YOLOv8
- **Speech**: Whisper (STT), Coqui (TTS)
- **Wake Word**: Picovoice Porcupine
- **Memory**: LangChain
- **Search**: DuckDuckGo API
- **Dashboard**: Streamlit
- **Languages**: Python

## 🏆 Why This Project Stands Out

- **Dual LLM Architecture** - Rare in student portfolios
- **Multi-Agent System** - Specialized agents for different tasks
- **Vision + Voice Integration** - True multimodal AI
- **Social Impact Focus** - Assistive technology for accessibility
- **Production-Ready Features** - Wake word, memory, multilingual, danger detection

---

**Built with ❤️ for accessibility and innovation**
