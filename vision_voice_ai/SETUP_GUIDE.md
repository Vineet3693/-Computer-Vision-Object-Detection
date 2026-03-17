# 🚀 Setup Guide - Vision Voice AI Assistant

## Prerequisites

- Python 3.9 or higher
- Webcam/microphone
- Google Gemini API key
- Groq API key

## Step 1: Clone/Download the Project

```bash
cd /workspace/vision_voice_ai
```

## Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** Installation may take 5-10 minutes due to large ML models (PyTorch, YOLOv8, etc.)

### Optional: Install Additional Audio Support

```bash
# For better audio playback
pip install sounddevice

# For wake word detection (optional)
pip install pvporcupine
```

## Step 4: Configure API Keys

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:

```env
# Get from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_actual_gemini_key_here

# Get from: https://console.groq.com/keys
GROQ_API_KEY=your_actual_groq_key_here
```

### How to Get API Keys

#### Google Gemini API Key
1. Go to https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and paste it in `.env`

#### Groq API Key
1. Go to https://console.groq.com/keys
2. Sign up or log in
3. Create a new API key
4. Copy the key and paste it in `.env`

**Both APIs have free tiers** suitable for development and testing!

## Step 5: Test the Installation

Run the component tests:

```bash
python examples_test.py
```

Expected output:
- ✅ Configuration status shown
- ✅ Chat agent test (if Groq key configured)
- ✅ Web agent test (DuckDuckGo search)
- ✅ Memory agent test
- ✅ Vision agent initialization
- ✅ Intent classification tests

## Step 6: Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Step 7: Using the Application

### In the Dashboard:

1. **Load AI Models** - Click the button in the sidebar
2. **Start Camera** - Enable camera feed
3. **Try Quick Actions**:
   - 👀 "What do I see?"
   - ⚠️ "Is it safe?"
4. **Type Commands** or use voice (with microphone)

### Voice Commands:

- "What do I see in front of me?" - Object detection
- "Where is the [object]?" - Locate specific item
- "Describe the scene" - Detailed analysis
- "Is it safe around me?" - Danger check
- "What's new?" - Scene changes
- General questions - Chat and web search

## Troubleshooting

### Common Issues

#### 1. "Module not found" errors
```bash
pip install -r requirements.txt --upgrade
```

#### 2. Camera not working
- Check camera index in `config.py` (try 0, 1, or 2)
- Ensure no other app is using the camera
- On Linux: `sudo chmod 666 /dev/video0`

#### 3. API errors
- Verify API keys in `.env` are correct
- Check internet connection
- Verify API quota hasn't been exceeded

#### 4. Slow performance
- Use smaller Whisper model: `WHISPER_MODEL=tiny`
- Reduce camera resolution in `config.py`
- Close other applications

#### 5. TTS not working
```bash
# Install system dependencies (Linux)
sudo apt-get install espeak

# Or install better TTS
pip install TTS
```

### Performance Optimization

For slower machines, edit `config.py`:

```python
WHISPER_MODEL = "tiny"  # Instead of "base"
YOLO_MODEL = "yolov8n.pt"  # Nano model (already default)
FRAME_WIDTH = 320  # Lower resolution
FRAME_HEIGHT = 240
```

## Project Structure

```
vision_voice_ai/
├── agents/           # Multi-agent system
│   ├── vision_agent.py    # YOLOv8 + Gemini Vision
│   ├── memory_agent.py    # LangChain memory
│   ├── chat_agent.py      # Groq + Llama3
│   └── web_agent.py       # DuckDuckGo search
├── core/             # Central orchestration
│   ├── orchestrator.py      # Intent detection & routing
│   └── response_generator.py # Natural responses
├── utils/            # Utilities
│   ├── speech_to_text.py    # Whisper STT
│   └── text_to_speech.py    # Coqui TTS
├── dashboard/        # Streamlit interface
│   └── app.py
├── config.py         # Configuration
├── requirements.txt  # Dependencies
└── examples_test.py  # Component tests
```

## Next Steps

### Basic Usage
1. Run the dashboard
2. Test object detection
3. Try conversation features
4. Experiment with different commands

### Advanced Customization
1. Add custom wake word
2. Train YOLOv8 on custom objects
3. Add more languages
4. Integrate additional APIs

### Deployment Options
- Run locally (current setup)
- Deploy to cloud (AWS, GCP, Azure)
- Package as desktop app
- Mobile integration

## Resources

- [Gemini AI Documentation](https://ai.google.dev/docs)
- [Groq API Documentation](https://console.groq.com/docs)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review error messages carefully
3. Verify all dependencies are installed
4. Ensure API keys are valid

---

**Happy Building! 🎉**

This project combines cutting-edge AI technologies to create an assistive tool that can make a real difference in people's lives. Enjoy exploring the possibilities!
