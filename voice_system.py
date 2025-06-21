#!/usr/bin/env python3
"""
Voice Synthesis System
Supports multiple TTS engines, voice cloning, and real-time audio streaming.
"""

import os
import logging
import asyncio
import tempfile
import wave
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
import torchaudio
from pathlib import Path
import soundfile as sf
import librosa
import webrtcvad
import pyaudio
from TTS.api import TTS
import pyttsx3
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRecorder
import websockets
import aiofiles
from datetime import datetime

logger = logging.getLogger(__name__)

class VoiceEngine(Enum):
    """Supported voice synthesis engines."""
    TTS = "tts"
    PYTTSX3 = "pyttsx3"
    COQUI = "coqui"
    ELEVENLABS = "elevenlabs"

class VoiceQuality(Enum):
    """Voice quality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

@dataclass
class VoiceConfig:
    """Configuration for voice synthesis."""
    engine: VoiceEngine = VoiceEngine.TTS
    voice_name: str = "tts_models/en/ljspeech/tacotron2-DDC"
    quality: VoiceQuality = VoiceQuality.MEDIUM
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    language: str = "en"
    sample_rate: int = 22050
    enable_voice_cloning: bool = False
    enable_emotion: bool = False
    enable_real_time: bool = False

class VoiceSynthesizer:
    """Main voice synthesis class supporting multiple engines."""
    
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.tts_engine = None
        self.pyttsx3_engine = None
        self.available_voices = {}
        self.current_voice = None
        self.audio_cache = {}
        self.voice_clones = {}
        
        # Initialize the selected engine
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the selected TTS engine."""
        try:
            if self.config.engine == VoiceEngine.TTS:
                self._initialize_tts()
            elif self.config.engine == VoiceEngine.PYTTSX3:
                self._initialize_pyttsx3()
            elif self.config.engine == VoiceEngine.COQUI:
                self._initialize_coqui()
            
            logger.info(f"Voice engine {self.config.engine.value} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice engine {self.config.engine.value}: {e}")
            # Fallback to pyttsx3
            self.config.engine = VoiceEngine.PYTTSX3
            self._initialize_pyttsx3()
    
    def _initialize_tts(self):
        """Initialize Coqui TTS engine."""
        try:
            # List available models
            available_models = TTS.list_models()
            logger.info(f"Available TTS models: {len(available_models)}")
            
            # Initialize TTS with selected model
            self.tts_engine = TTS(model_name=self.config.voice_name)
            
            # Get available voices
            if hasattr(self.tts_engine, 'speakers'):
                self.available_voices = {speaker: speaker for speaker in self.tts_engine.speakers}
            
            logger.info(f"TTS engine initialized with model: {self.config.voice_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            raise
    
    def _initialize_pyttsx3(self):
        """Initialize pyttsx3 engine."""
        try:
            self.pyttsx3_engine = pyttsx3.init()
            
            # Configure voice properties
            self.pyttsx3_engine.setProperty('rate', int(200 * self.config.speed))
            self.pyttsx3_engine.setProperty('volume', self.config.volume)
            
            # Get available voices
            voices = self.pyttsx3_engine.getProperty('voices')
            self.available_voices = {voice.name: voice.id for voice in voices}
            
            # Set default voice
            if voices:
                self.pyttsx3_engine.setProperty('voice', voices[0].id)
                self.current_voice = voices[0].id
            
            logger.info(f"pyttsx3 engine initialized with {len(voices)} voices")
            
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3: {e}")
            raise
    
    def _initialize_coqui(self):
        """Initialize Coqui TTS with advanced features."""
        try:
            # Use Coqui TTS with advanced configuration
            self.tts_engine = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
                gpu=torch.cuda.is_available()
            )
            
            logger.info("Coqui TTS with XTTS v2 initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Coqui TTS: {e}")
            raise
    
    def synthesize_text(self, text: str, voice_name: Optional[str] = None, 
                       emotion: Optional[str] = None) -> bytes:
        """Synthesize text to speech and return audio data."""
        try:
            # Check cache first
            cache_key = f"{text}_{voice_name}_{emotion}_{self.config.engine.value}"
            if cache_key in self.audio_cache:
                logger.info("Using cached audio")
                return self.audio_cache[cache_key]
            
            if self.config.engine == VoiceEngine.TTS:
                audio_data = self._synthesize_tts(text, voice_name, emotion)
            elif self.config.engine == VoiceEngine.PYTTSX3:
                audio_data = self._synthesize_pyttsx3(text, voice_name)
            elif self.config.engine == VoiceEngine.COQUI:
                audio_data = self._synthesize_coqui(text, voice_name, emotion)
            else:
                raise ValueError(f"Unsupported engine: {self.config.engine}")
            
            # Cache the result
            self.audio_cache[cache_key] = audio_data
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Failed to synthesize text: {e}")
            return b""
    
    def _synthesize_tts(self, text: str, voice_name: Optional[str], emotion: Optional[str]) -> bytes:
        """Synthesize using Coqui TTS."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Generate speech
            self.tts_engine.tts_to_file(
                text=text,
                file_path=temp_path,
                speaker=voice_name if voice_name else None,
                language=self.config.language
            )
            
            # Read the generated audio
            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up
            os.unlink(temp_path)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return b""
    
    def _synthesize_pyttsx3(self, text: str, voice_name: Optional[str]) -> bytes:
        """Synthesize using pyttsx3."""
        try:
            # Set voice if specified
            if voice_name and voice_name in self.available_voices:
                self.pyttsx3_engine.setProperty('voice', self.available_voices[voice_name])
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Generate speech
            self.pyttsx3_engine.save_to_file(text, temp_path)
            self.pyttsx3_engine.runAndWait()
            
            # Read the generated audio
            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up
            os.unlink(temp_path)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"pyttsx3 synthesis failed: {e}")
            return b""
    
    def _synthesize_coqui(self, text: str, voice_name: Optional[str], emotion: Optional[str]) -> bytes:
        """Synthesize using Coqui TTS with advanced features."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Generate speech with emotion if supported
            if emotion and hasattr(self.tts_engine, 'tts_to_file_with_emotion'):
                self.tts_engine.tts_to_file_with_emotion(
                    text=text,
                    file_path=temp_path,
                    emotion=emotion,
                    language=self.config.language
                )
            else:
                self.tts_engine.tts_to_file(
                    text=text,
                    file_path=temp_path,
                    language=self.config.language
                )
            
            # Read the generated audio
            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up
            os.unlink(temp_path)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Coqui synthesis failed: {e}")
            return b""
    
    def synthesize_to_file(self, text: str, output_path: str, 
                          voice_name: Optional[str] = None, 
                          emotion: Optional[str] = None) -> bool:
        """Synthesize text to speech and save to file."""
        try:
            audio_data = self.synthesize_text(text, voice_name, emotion)
            
            if audio_data:
                with open(output_path, 'wb') as f:
                    f.write(audio_data)
                logger.info(f"Audio saved to: {output_path}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to save audio to file: {e}")
            return False
    
    def get_available_voices(self) -> Dict[str, str]:
        """Get available voices for the current engine."""
        return self.available_voices.copy()
    
    def set_voice(self, voice_name: str) -> bool:
        """Set the current voice."""
        try:
            if voice_name in self.available_voices:
                if self.config.engine == VoiceEngine.PYTTSX3:
                    self.pyttsx3_engine.setProperty('voice', self.available_voices[voice_name])
                    self.current_voice = self.available_voices[voice_name]
                else:
                    self.current_voice = voice_name
                
                logger.info(f"Voice set to: {voice_name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to set voice: {e}")
            return False
    
    def clone_voice(self, voice_name: str, audio_samples: List[str]) -> bool:
        """Clone a voice using provided audio samples."""
        if not self.config.enable_voice_cloning:
            logger.warning("Voice cloning is not enabled")
            return False
        
        try:
            # This is a simplified voice cloning implementation
            # In production, you would use a proper voice cloning model
            logger.info(f"Voice cloning for {voice_name} with {len(audio_samples)} samples")
            
            # Store the voice clone configuration
            self.voice_clones[voice_name] = {
                "samples": audio_samples,
                "created_at": datetime.now().isoformat()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            return False
    
    def clear_cache(self):
        """Clear the audio cache."""
        self.audio_cache.clear()
        logger.info("Audio cache cleared")
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the current engine."""
        return {
            "engine": self.config.engine.value,
            "voice_name": self.config.voice_name,
            "quality": self.config.quality.value,
            "speed": self.config.speed,
            "pitch": self.config.pitch,
            "volume": self.config.volume,
            "language": self.config.language,
            "sample_rate": self.config.sample_rate,
            "available_voices": len(self.available_voices),
            "current_voice": self.current_voice,
            "cache_size": len(self.audio_cache),
            "voice_clones": len(self.voice_clones)
        }

class RealTimeVoiceStream:
    """Real-time voice streaming for live conversations."""
    
    def __init__(self, synthesizer: VoiceSynthesizer):
        self.synthesizer = synthesizer
        self.is_streaming = False
        self.audio_queue = asyncio.Queue()
        self.websocket_server = None
        self.clients = set()
    
    async def start_streaming(self, host: str = "localhost", port: int = 8765):
        """Start the real-time voice streaming server."""
        try:
            self.websocket_server = await websockets.serve(
                self._handle_client, host, port
            )
            self.is_streaming = True
            logger.info(f"Voice streaming server started on ws://{host}:{port}")
            
        except Exception as e:
            logger.error(f"Failed to start voice streaming: {e}")
    
    async def stop_streaming(self):
        """Stop the real-time voice streaming server."""
        self.is_streaming = False
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        logger.info("Voice streaming server stopped")
    
    async def _handle_client(self, websocket, path):
        """Handle WebSocket client connections."""
        self.clients.add(websocket)
        try:
            async for message in websocket:
                await self._process_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
    
    async def _process_message(self, websocket, message):
        """Process incoming WebSocket messages."""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "synthesize":
                text = data.get("text", "")
                voice_name = data.get("voice_name")
                emotion = data.get("emotion")
                
                # Synthesize audio
                audio_data = self.synthesizer.synthesize_text(text, voice_name, emotion)
                
                # Send audio data back to client
                await websocket.send(json.dumps({
                    "type": "audio",
                    "data": audio_data.hex(),
                    "sample_rate": self.synthesizer.config.sample_rate
                }))
            
            elif message_type == "set_voice":
                voice_name = data.get("voice_name")
                success = self.synthesizer.set_voice(voice_name)
                await websocket.send(json.dumps({
                    "type": "voice_set",
                    "success": success
                }))
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": str(e)
            }))
    
    async def broadcast_audio(self, text: str, voice_name: Optional[str] = None):
        """Broadcast synthesized audio to all connected clients."""
        if not self.clients:
            return
        
        try:
            audio_data = self.synthesizer.synthesize_text(text, voice_name)
            
            message = json.dumps({
                "type": "broadcast_audio",
                "data": audio_data.hex(),
                "sample_rate": self.synthesizer.config.sample_rate
            })
            
            # Send to all connected clients
            await asyncio.gather(*[
                client.send(message) for client in self.clients
            ], return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Failed to broadcast audio: {e}")

class VoiceRecorder:
    """Voice recording and processing for voice commands."""
    
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.audio_frames = []
    
    def start_recording(self) -> bool:
        """Start recording audio."""
        try:
            self.audio_frames = []
            self.is_recording = True
            
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            logger.info("Voice recording started")
            
            while self.is_recording:
                data = stream.read(self.chunk_size)
                self.audio_frames.append(data)
                
                # Check for voice activity
                if self.vad.is_speech(data, self.sample_rate):
                    # Continue recording while voice is detected
                    pass
            
            stream.stop_stream()
            stream.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return False
    
    def stop_recording(self) -> bytes:
        """Stop recording and return audio data."""
        self.is_recording = False
        
        if self.audio_frames:
            # Combine all audio frames
            audio_data = b''.join(self.audio_frames)
            logger.info(f"Recording stopped, captured {len(audio_data)} bytes")
            return audio_data
        
        return b""
    
    def save_recording(self, filepath: str) -> bool:
        """Save the recorded audio to a file."""
        try:
            audio_data = self.stop_recording()
            
            if audio_data:
                with wave.open(filepath, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(audio_data)
                
                logger.info(f"Recording saved to: {filepath}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to save recording: {e}")
            return False
    
    def cleanup(self):
        """Clean up audio resources."""
        self.audio.terminate()

# Global voice system instance
_voice_system = None

def get_voice_system(config: Optional[VoiceConfig] = None) -> VoiceSynthesizer:
    """Get the global voice system instance."""
    global _voice_system
    if _voice_system is None:
        if config is None:
            config = VoiceConfig()
        _voice_system = VoiceSynthesizer(config)
    return _voice_system

def get_voice_stream(synthesizer: Optional[VoiceSynthesizer] = None) -> RealTimeVoiceStream:
    """Get a real-time voice stream instance."""
    if synthesizer is None:
        synthesizer = get_voice_system()
    return RealTimeVoiceStream(synthesizer)

def get_voice_recorder() -> VoiceRecorder:
    """Get a voice recorder instance."""
    return VoiceRecorder() 