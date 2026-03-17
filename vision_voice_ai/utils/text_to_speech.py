"""
Text to Speech - Converts AI responses to spoken audio using Coqui TTS
Supports multiple languages and voices
"""
import numpy as np
from typing import Optional
import tempfile
import os

from config import TTS_SPEAKER, TTS_SPEED, DEFAULT_LANGUAGE


class TextToSpeech:
    """
    Text to Speech converter using Coqui TTS
    
    Features:
    - Natural-sounding voices
    - Multi-language support
    - Adjustable speed and pitch
    - Real-time synthesis
    """

    def __init__(self, language: str = None):
        self.language = language or DEFAULT_LANGUAGE
        self.speaker = TTS_SPEAKER
        self.speed = TTS_SPEED
        
        # Initialize TTS model
        print("Initializing Coqui TTS...")
        try:
            from TTS.api import TTS
            
            # Select appropriate model based on language
            if self.language == "hi":
                # Hindi model
                self.model_name = "tts_models/multilingual/multi-dataset/your_tts"
            else:
                # English model (better quality)
                self.model_name = "tts_models/en/ljspeech/tacotron2-DDC"
            
            self.tts = TTS(model_name=self.model_name, progress_bar=False)
            print("Coqui TTS initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing Coqui TTS: {e}")
            print("Falling back to simple TTS...")
            self.tts = None

    def synthesize(self, text: str, 
                   output_file: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Convert text to speech
        
        Args:
            text: Text to synthesize
            output_file: Optional output file path
            
        Returns:
            Audio data as numpy array, or None if output_file specified
        """
        if not self.tts:
            return self._fallback_synthesize(text, output_file)
        
        try:
            if output_file:
                # Save to file
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_file,
                    speaker=self.speaker,
                    speed=self.speed
                )
                return None
            else:
                # Return audio data
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    self.tts.tts_to_file(
                        text=text,
                        file_path=tmp_file.name,
                        speaker=self.speaker,
                        speed=self.speed
                    )
                    
                    # Read audio data
                    import wave
                    with wave.open(tmp_file.name, 'rb') as wav_file:
                        frames = wav_file.readframes(wav_file.getnframes())
                        audio_data = np.frombuffer(frames, dtype=np.int16)
                    
                    os.unlink(tmp_file.name)
                    return audio_data
                    
        except Exception as e:
            print(f"TTS synthesis error: {e}")
            return None

    def speak(self, text: str, block: bool = True):
        """
        Synthesize and play audio immediately
        
        Args:
            text: Text to speak
            block: Whether to block until playback completes
        """
        audio_data = self.synthesize(text)
        
        if audio_data is not None:
            self._play_audio(audio_data, block)

    def _play_audio(self, audio_data: np.ndarray, block: bool = True):
        """Play audio data through speakers"""
        try:
            import sounddevice as sd
            
            # Play audio
            sd.play(audio_data, samplerate=22050, block=block)
            
            if not block:
                return
            
            # Wait for playback to complete
            sd.wait()
            
        except Exception as e:
            print(f"Audio playback error: {e}")
            print("Consider installing sounddevice: pip install sounddevice")

    def _fallback_synthesize(self, text: str, 
                            output_file: Optional[str] = None) -> Optional[np.ndarray]:
        """Fallback TTS using system commands"""
        print("Using fallback TTS...")
        
        try:
            # Try using system TTS
            if os.name == 'nt':  # Windows
                import subprocess
                if output_file:
                    # Can't easily save on Windows without additional libs
                    subprocess.run(['powershell', '-Command', 
                                  f'Add-Type -AssemblyName System.Speech; ' +
                                  f'$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; ' +
                                  f'$speak.Speak("{text}")'])
                else:
                    subprocess.run(['powershell', '-Command',
                                  f'Add-Type -AssemblyName System.Speech; ' +
                                  f'$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; ' +
                                  f'$speak.Speak("{text}")'])
            
            elif os.name == 'posix':  # Linux/Mac
                if output_file:
                    os.system(f'espeak -w "{output_file}" "{text}"')
                else:
                    os.system(f'espeak "{text}"')
            
            return None
            
        except Exception as e:
            print(f"Fallback TTS error: {e}")
            return None

    def set_language(self, language: str):
        """Set TTS language"""
        supported = ["en", "hi", "es", "fr", "de", "it", "pt", "ja", "zh"]
        if language in supported:
            self.language = language
            print(f"TTS language set to: {language}")
            
            # Reinitialize with new language
            if self.tts:
                try:
                    from TTS.api import TTS
                    if language == "hi":
                        self.model_name = "tts_models/multilingual/multi-dataset/your_tts"
                    else:
                        self.model_name = "tts_models/en/ljspeech/tacotron2-DDC"
                    
                    self.tts = TTS(model_name=self.model_name, progress_bar=False)
                except Exception as e:
                    print(f"Error reloading TTS: {e}")
        else:
            print(f"Unsupported language. Choose from: {supported}")

    def set_speed(self, speed: float):
        """Set speech speed (0.5 = slow, 2.0 = fast)"""
        if 0.5 <= speed <= 2.0:
            self.speed = speed
            print(f"TTS speed set to: {speed}x")
        else:
            print("Speed must be between 0.5 and 2.0")

    def get_available_voices(self) -> list:
        """Get list of available voices"""
        if not self.tts:
            return ["default"]
        
        try:
            return self.tts.speakers
        except Exception:
            return ["default"]

    def set_voice(self, speaker: str):
        """Set voice/speaker"""
        self.speaker = speaker
        print(f"Voice set to: {speaker}")
