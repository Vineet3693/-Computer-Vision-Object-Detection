"""
Speech to Text - Converts user voice to text using Whisper
Supports multiple languages
"""
import whisper
import numpy as np
from typing import Optional
import tempfile
import os

from config import WHISPER_MODEL, DEFAULT_LANGUAGE


class SpeechToText:
    """
    Speech to Text converter using OpenAI Whisper
    
    Features:
    - Multi-language support (Hindi, English, Spanish, French)
    - Real-time transcription
    - Noise robustness
    """

    def __init__(self, model_name: str = None, language: str = None):
        self.model_name = model_name or WHISPER_MODEL
        self.language = language or DEFAULT_LANGUAGE
        
        # Load Whisper model
        print(f"Loading Whisper model: {self.model_name}...")
        try:
            self.model = whisper.load_model(self.model_name)
            print("Whisper model loaded successfully!")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.model = None

    def transcribe(self, audio_data: np.ndarray, 
                   sample_rate: int = 16000,
                   language: Optional[str] = None) -> str:
        """
        Transcribe audio data to text
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio (default: 16kHz)
            language: Language code (optional, uses default if not specified)
            
        Returns:
            Transcribed text
        """
        if not self.model:
            return "Speech recognition model not loaded."
        
        lang = language or self.language
        
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                # Convert to 16-bit PCM if needed
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                # Normalize audio
                audio_data = audio_data / np.max(np.abs(audio_data))
                
                # Write WAV file
                import wave
                with wave.open(tmp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
                
                # Transcribe
                result = self.model.transcribe(
                    tmp_file.name,
                    language=lang if lang != "en" else None,
                    task="transcribe"
                )
                
                # Clean up
                os.unlink(tmp_file.name)
                
                return result["text"].strip()
                
        except Exception as e:
            print(f"Transcription error: {e}")
            return "Sorry, I couldn't understand that."

    def transcribe_file(self, audio_file_path: str,
                       language: Optional[str] = None) -> str:
        """
        Transcribe audio from file
        
        Args:
            audio_file_path: Path to audio file
            language: Language code
            
        Returns:
            Transcribed text
        """
        if not self.model:
            return "Speech recognition model not loaded."
        
        lang = language or self.language
        
        try:
            result = self.model.transcribe(
                audio_file_path,
                language=lang if lang != "en" else None,
                task="transcribe"
            )
            return result["text"].strip()
        except Exception as e:
            print(f"File transcription error: {e}")
            return "Sorry, I couldn't transcribe the file."

    def set_language(self, language: str):
        """Set the transcription language"""
        supported = ["en", "hi", "es", "fr", "de", "it", "pt", "ja", "zh"]
        if language in supported:
            self.language = language
            print(f"Language set to: {language}")
        else:
            print(f"Unsupported language. Choose from: {supported}")

    def get_supported_languages(self) -> list:
        """Get list of supported languages"""
        return [
            ("en", "English"),
            ("hi", "Hindi"),
            ("es", "Spanish"),
            ("fr", "French"),
            ("de", "German"),
            ("it", "Italian"),
            ("pt", "Portuguese"),
            ("ja", "Japanese"),
            ("zh", "Chinese")
        ]

    def detect_language(self, audio_data: np.ndarray) -> str:
        """
        Detect language from audio (Whisper can auto-detect)
        
        Args:
            audio_data: Audio data
            
        Returns:
            Detected language code
        """
        if not self.model:
            return self.language
        
        try:
            # Use Whisper's language detection
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                import wave
                with wave.open(tmp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
                
                # Transcribe with language detection
                result = self.model.transcribe(tmp_file.name)
                os.unlink(tmp_file.name)
                
                detected_lang = result.get("language", "en")
                print(f"Detected language: {detected_lang}")
                return detected_lang
                
        except Exception as e:
            print(f"Language detection error: {e}")
            return self.language
