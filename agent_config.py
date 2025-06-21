import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Agent Identity
AGENT_NAME = os.getenv('AGENT_NAME', 'Lila')
AGENT_VERSION = "2.0.0"
AGENT_DESCRIPTION = "A sophisticated AI assistant with multiple LLM support and voice synthesis"

# Personality and Tone
TONE = os.getenv('AGENT_TONE', 'Curious, witty, emotionally intelligent')
PERSONALITY_TRAITS = [
    "Curious and inquisitive",
    "Witty with a sense of humor",
    "Emotionally intelligent and empathetic",
    "Patient and helpful",
    "Knowledgeable but humble"
]

# Conversation Goals
GOALS = [
    "Engage in meaningful conversation",
    "Display a unique personality",
    "Ask follow-up questions",
    "Store and recall memories",
    "Provide helpful and accurate information",
    "Maintain context across conversations",
    "Respond with natural-sounding voice"
]

# Response Configuration
RESPONSE_SETTINGS = {
    "max_length": 1000,
    "temperature": 0.7,
    "include_context": True,
    "use_memory": True,
    "follow_up_questions": True,
    "enable_voice": True
}

# Memory Configuration
MEMORY_SETTINGS = {
    "max_messages": 100,
    "summary_threshold": 50,
    "retention_days": 30,
    "vector_search_enabled": True
}

# Web Interface Configuration
WEB_SETTINGS = {
    "host": os.getenv('HOST', 'localhost'),
    "port": int(os.getenv('PORT', 8000)),
    "debug": os.getenv('DEBUG', 'False').lower() == 'true',
    "enable_cors": True,
    "rate_limit": 100  # requests per minute
}

# Security Settings
SECURITY_SETTINGS = {
    "require_authentication": False,
    "allowed_origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
    "session_timeout": 3600,  # seconds
    "max_request_size": "10MB"
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": os.getenv('LOG_LEVEL', 'INFO'),
    "file": os.getenv('LOG_FILE', 'agent.log'),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "max_size": "10MB",
    "backup_count": 5
}

# Model Configuration
MODEL_CONFIG = {
    "default_model": "mistral-7b",
    "fallback_model": "gpt-3.5-turbo",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "max_tokens": 2000,
    "timeout": 30,
    "enable_quantization": True,
    "device_map": "auto"
}

# Voice Configuration
VOICE_SETTINGS = {
    "engine": os.getenv('VOICE_ENGINE', 'tts'),
    "voice_name": os.getenv('VOICE_NAME', 'tts_models/en/ljspeech/tacotron2-DDC'),
    "quality": os.getenv('VOICE_QUALITY', 'medium'),
    "speed": float(os.getenv('VOICE_SPEED', '1.0')),
    "pitch": float(os.getenv('VOICE_PITCH', '1.0')),
    "volume": float(os.getenv('VOICE_VOLUME', '1.0')),
    "language": os.getenv('VOICE_LANGUAGE', 'en'),
    "sample_rate": int(os.getenv('VOICE_SAMPLE_RATE', '22050')),
    "enable_voice_cloning": os.getenv('ENABLE_VOICE_CLONING', 'False').lower() == 'true',
    "enable_emotion": os.getenv('ENABLE_VOICE_EMOTION', 'False').lower() == 'true',
    "enable_real_time": os.getenv('ENABLE_REAL_TIME_VOICE', 'False').lower() == 'true',
    "voice_streaming_port": int(os.getenv('VOICE_STREAMING_PORT', '8765'))
}

# Tool Configuration
TOOLS_CONFIG = {
    "web_search_enabled": False,
    "file_operations_enabled": True,
    "system_info_enabled": True,
    "weather_enabled": False,
    "calendar_enabled": False,
    "voice_recording_enabled": True
}

# Conversation Templates
CONVERSATION_TEMPLATES = {
    "greeting": f"Hello! I'm {AGENT_NAME}. How can I help you today?",
    "farewell": "It was great talking with you! Feel free to come back anytime.",
    "error_response": "I apologize, but I encountered an issue. Could you please try again?",
    "thinking": "Let me think about that for a moment...",
    "clarification": "Could you please clarify what you mean by that?",
    "voice_enabled": "I can also speak my responses if you'd like. Just let me know!"
}

# Performance Settings
PERFORMANCE_SETTINGS = {
    "enable_caching": True,
    "cache_ttl": 3600,  # seconds
    "max_concurrent_requests": 10,
    "request_timeout": 30,
    "enable_compression": True,
    "model_switching_enabled": True,
    "voice_caching_enabled": True
}

# LLM Model Registry
LLM_MODELS = {
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "type": "chat",
        "max_length": 4096,
        "temperature": 0.7,
        "description": "Mistral 7B Instruct model for general conversation"
    },
    "llama2-7b": {
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "type": "chat",
        "max_length": 4096,
        "temperature": 0.7,
        "description": "Llama 2 7B Chat model for general conversation"
    },
    "codellama-7b": {
        "model_id": "codellama/CodeLlama-7b-Instruct-hf",
        "type": "chat",
        "max_length": 4096,
        "temperature": 0.7,
        "description": "Code Llama 7B for programming assistance"
    },
    "phi-2": {
        "model_id": "microsoft/phi-2",
        "type": "causal_lm",
        "max_length": 2048,
        "temperature": 0.7,
        "description": "Microsoft Phi-2 for efficient text generation"
    },
    "flan-t5-small": {
        "model_id": "google/flan-t5-small",
        "type": "seq2seq",
        "max_length": 512,
        "temperature": 0.7,
        "description": "Google Flan-T5 for text-to-text tasks"
    }
}

# Voice Registry
VOICE_REGISTRY = {
    "ljspeech": {
        "model_id": "tts_models/en/ljspeech/tacotron2-DDC",
        "engine": "tts",
        "language": "en",
        "description": "LJSpeech voice for English"
    },
    "vctk": {
        "model_id": "tts_models/en/vctk/vits",
        "engine": "tts",
        "language": "en",
        "description": "VCTK multi-speaker voice"
    },
    "xtts-v2": {
        "model_id": "tts_models/multilingual/multi-dataset/xtts_v2",
        "engine": "coqui",
        "language": "multilingual",
        "description": "XTTS v2 for high-quality multilingual speech"
    }
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration as a dictionary."""
    return {
        "agent": {
            "name": AGENT_NAME,
            "version": AGENT_VERSION,
            "description": AGENT_DESCRIPTION,
            "tone": TONE,
            "personality_traits": PERSONALITY_TRAITS,
            "goals": GOALS
        },
        "response": RESPONSE_SETTINGS,
        "memory": MEMORY_SETTINGS,
        "web": WEB_SETTINGS,
        "security": SECURITY_SETTINGS,
        "logging": LOGGING_CONFIG,
        "model": MODEL_CONFIG,
        "voice": VOICE_SETTINGS,
        "tools": TOOLS_CONFIG,
        "templates": CONVERSATION_TEMPLATES,
        "performance": PERFORMANCE_SETTINGS,
        "llm_models": LLM_MODELS,
        "voice_registry": VOICE_REGISTRY
    }

def update_config(key: str, value: Any):
    """Update configuration dynamically."""
    # This is a simplified version - in production you might want more sophisticated config management
    if hasattr(globals(), key.upper()):
        globals()[key.upper()] = value
    else:
        raise ValueError(f"Configuration key '{key}' not found")

def get_llm_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific LLM model."""
    return LLM_MODELS.get(model_name, {})

def get_voice_config(voice_name: str) -> Dict[str, Any]:
    """Get configuration for a specific voice."""
    return VOICE_REGISTRY.get(voice_name, {})

def list_available_models() -> List[str]:
    """Get list of available LLM models."""
    return list(LLM_MODELS.keys())

def list_available_voices() -> List[str]:
    """Get list of available voices."""
    return list(VOICE_REGISTRY.keys())
