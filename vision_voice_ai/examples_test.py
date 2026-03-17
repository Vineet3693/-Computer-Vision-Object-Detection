"""
Example script demonstrating the Vision Voice AI system
Run this to test individual components
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import GEMINI_API_KEY, GROQ_API_KEY
from core.orchestrator import MasterOrchestrator
from agents.vision_agent import VisionAgent
from agents.chat_agent import ChatAgent
from agents.web_agent import WebAgent
from agents.memory_agent import MemoryAgent


def test_chat_agent():
    """Test the Chat Agent with Groq"""
    print("\n" + "="*60)
    print("Testing Chat Agent (Groq + Llama3)")
    print("="*60)
    
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        print("⚠️  Groq API key not configured. Skipping test.")
        return
    
    chat_agent = ChatAgent()
    
    # Test general conversation
    questions = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms",
        "What's the weather like today?"
    ]
    
    for question in questions:
        print(f"\n❓ Q: {question}")
        response = chat_agent.answer_question(question)
        print(f"💬 A: {response}")


def test_web_agent():
    """Test the Web Agent with DuckDuckGo"""
    print("\n" + "="*60)
    print("Testing Web Agent (DuckDuckGo Search)")
    print("="*60)
    
    web_agent = WebAgent()
    
    queries = [
        "What is artificial intelligence?",
        "Latest AI news",
        "How does photosynthesis work?"
    ]
    
    for query in queries:
        print(f"\n🔍 Search: {query}")
        results = web_agent.search(query, max_results=2)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.get('title', 'No title')}")
                print(f"     {result.get('body', 'No description')[:150]}...")
        else:
            print("  No results found")


def test_memory_agent():
    """Test the Memory Agent"""
    print("\n" + "="*60)
    print("Testing Memory Agent (LangChain)")
    print("="*60)
    
    memory_agent = MemoryAgent()
    
    # Simulate conversation
    conversations = [
        ("Hello, what can you do?", 
         "I'm a vision assistant that can describe your surroundings, detect objects, and help you navigate safely."),
        ("What objects do I have?",
         "I can see a laptop, a water bottle, and a notebook on your desk."),
        ("Where is the bottle?",
         "The water bottle is on your right side, next to the laptop.")
    ]
    
    print("\n📝 Storing conversations in memory...")
    for user_msg, ai_msg in conversations:
        memory_agent.add_conversation_pair(user_msg, ai_msg)
        print(f"  ✓ Added: {user_msg[:40]}...")
    
    # Test memory recall
    print("\n🧠 Testing memory recall...")
    summary = memory_agent.get_session_summary()
    print(f"  Session Duration: {summary['session_duration']}")
    print(f"  Total Interactions: {summary['total_interactions']}")
    print(f"  Conversation Turns: {summary['conversation_turns']}")
    
    # Get conversation history
    print("\n📜 Recent Conversation History:")
    history = memory_agent.get_conversation_history()
    print(history)


def test_vision_agent():
    """Test the Vision Agent"""
    print("\n" + "="*60)
    print("Testing Vision Agent (YOLOv8 + Gemini Vision)")
    print("="*60)
    
    vision_agent = VisionAgent()
    
    print(f"\n✅ YOLOv8 model loaded: {vision_agent.yolo_model is not None}")
    print(f"✅ Gemini Vision available: {vision_agent.vision_model is not None}")
    print(f"📦 Detectable object classes: {len(vision_agent.class_names)}")
    
    # Show some example classes
    print("\n🎯 Example detectable objects:")
    sample_objects = ['person', 'car', 'laptop', 'bottle', 'chair', 'book']
    for obj in sample_objects:
        if obj in vision_agent.class_names.values():
            print(f"  ✓ {obj}")
    
    print("\n💡 Note: To test actual object detection, run the Streamlit dashboard")
    print("   with a camera connected.")


def test_orchestrator():
    """Test the Master Orchestrator"""
    print("\n" + "="*60)
    print("Testing Master Orchestrator (Intent Detection & Routing)")
    print("="*60)
    
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        print("⚠️  Groq API key not configured. Using rule-based classification.")
    
    orchestrator = MasterOrchestrator()
    
    # Test intent classification
    test_queries = [
        "What do I see in front of me?",
        "Where is my laptop?",
        "Is it safe around here?",
        "Describe the scene",
        "What's the capital of Germany?",
        "Tell me about yourself",
        "Do you remember what I asked before?"
    ]
    
    print("\n🧠 Intent Classification Tests:")
    for query in test_queries:
        intent_result = orchestrator.classify_intent(query)
        print(f"\n  Query: \"{query}\"")
        print(f"  Intent: {intent_result['intent']}")
        print(f"  Confidence: {intent_result['confidence']:.2f}")
        print(f"  Requires Vision: {intent_result['requires_vision']}")
        print(f"  Requires Web: {intent_result.get('requires_web', False)}")


def main():
    """Run all tests"""
    print("\n" + "🚀"*30)
    print("Vision Voice AI - Component Tests")
    print("🚀"*30)
    
    # Check API keys
    print("\n📋 Configuration Status:")
    print(f"  Gemini API Key: {'✅ Configured' if GEMINI_API_KEY and GEMINI_API_KEY != 'your_gemini_api_key_here' else '❌ Not configured'}")
    print(f"  Groq API Key: {'✅ Configured' if GROQ_API_KEY and GROQ_API_KEY != 'your_groq_api_key_here' else '❌ Not configured'}")
    
    # Run tests
    test_chat_agent()
    test_web_agent()
    test_memory_agent()
    test_vision_agent()
    test_orchestrator()
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)
    print("\n📌 Next Steps:")
    print("  1. Configure your API keys in .env file")
    print("  2. Run: streamlit run dashboard/app.py")
    print("  3. Use voice commands or text input to interact")
    print("\n🎤 Example Voice Commands:")
    print('  - "What do I see?"')
    print('  - "Where is the bottle?"')
    print('  - "Is it safe?"')
    print('  - "Describe the scene"')
    print('  - "What\'s the latest AI news?"')
    print()


if __name__ == "__main__":
    main()
