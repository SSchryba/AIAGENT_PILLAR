import os
import logging
import hashlib
import time
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from memory import get_memory
from llm_manager import get_llm_manager, HuggingFaceLLM
from voice_system import get_voice_system, VoiceConfig, VoiceEngine
import agent_config as config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.getenv('LOG_FILE', 'agent.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ResponseCache:
    """Simple in-memory cache for responses with TTL."""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def _get_cache_key(self, message: str) -> str:
        """Generate a cache key for a message."""
        return hashlib.md5(message.encode()).hexdigest()
    
    def get(self, message: str) -> Optional[str]:
        """Get a cached response if available and not expired."""
        key = self._get_cache_key(message)
        if key in self.cache:
            timestamp, response = self.cache[key]
            if time.time() - timestamp < self.ttl:
                logger.info(f"Cache hit for message: {message[:50]}...")
                return response
            else:
                del self.cache[key]
        return None
    
    def set(self, message: str, response: str):
        """Cache a response with current timestamp."""
        key = self._get_cache_key(message)
        self.cache[key] = (time.time(), response)
        logger.info(f"Cached response for message: {message[:50]}...")
    
    def clear(self):
        """Clear all cached responses."""
        self.cache.clear()
        logger.info("Response cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache),
            "ttl_seconds": self.ttl
        }

class AIAgent:
    """Enhanced AI Agent with multiple LLM support and voice synthesis."""
    
    def __init__(self, model_name: str = "mistral-7b", temperature: float = 0.7, 
                 enable_voice: bool = True, voice_engine: VoiceEngine = VoiceEngine.TTS):
        self.model_name = model_name
        self.temperature = temperature
        self.enable_voice = enable_voice
        self.memory = None
        self.conversation = None
        self.cache = ResponseCache(ttl_seconds=config.PERFORMANCE_SETTINGS.get("cache_ttl", 3600))
        self.llm_manager = None
        self.voice_system = None
        
        # Initialize components
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the agent with memory, LLM manager, and voice system."""
        try:
            # Initialize memory
            self.memory = get_memory()
            
            # Initialize LLM manager
            self.llm_manager = get_llm_manager()
            
            # Load the specified model
            if not self.llm_manager.load_model(self.model_name):
                logger.warning(f"Failed to load model {self.model_name}, trying default model")
                available_models = self.llm_manager.get_available_models()
                if available_models:
                    self.model_name = available_models[0]
                    self.llm_manager.switch_model(self.model_name)
                else:
                    raise RuntimeError("No LLM models available")
            
            # Initialize voice system if enabled
            if self.enable_voice:
                voice_config = VoiceConfig(
                    engine=VoiceEngine.TTS,
                    voice_name="tts_models/en/ljspeech/tacotron2-DDC",
                    quality=config.VOICE_SETTINGS.get("quality", "medium"),
                    speed=config.VOICE_SETTINGS.get("speed", 1.0),
                    volume=config.VOICE_SETTINGS.get("volume", 1.0),
                    language=config.VOICE_SETTINGS.get("language", "en"),
                    enable_voice_cloning=config.VOICE_SETTINGS.get("enable_voice_cloning", False),
                    enable_emotion=config.VOICE_SETTINGS.get("enable_emotion", False)
                )
                self.voice_system = get_voice_system(voice_config)
            
            # Create custom prompt template
            template = f"""You are {config.AGENT_NAME}, an AI assistant with the following characteristics:
            - Tone: {config.TONE}
            - Goals: {', '.join(config.GOALS)}
            
            Current conversation:
            {{history}}
            Human: {{input}}
            {config.AGENT_NAME}:"""
            
            prompt = PromptTemplate(
                input_variables=["history", "input"],
                template=template
            )
            
            # Initialize conversation chain with Hugging Face LLM
            try:
                llm = HuggingFaceLLM(model_name=self.model_name)
                self.conversation = ConversationChain(
                    llm=llm,
                    verbose=os.getenv('DEBUG', 'False').lower() == 'true',
                    memory=self.memory,
                    prompt=prompt
                )
            except Exception as e:
                logger.warning(f"Hugging Face LLM initialization failed: {e}")
                # Fallback to OpenAI if available
                try:
                    llm = OpenAI(
                        model_name="gpt-3.5-turbo",
                        temperature=self.temperature,
                        openai_api_key=os.getenv('OPENAI_API_KEY')
                    )
                    self.conversation = ConversationChain(
                        llm=llm,
                        verbose=os.getenv('DEBUG', 'False').lower() == 'true',
                        memory=self.memory,
                        prompt=prompt
                    )
                except Exception as e2:
                    logger.error(f"All LLM initialization failed: {e2}")
                    raise
            
            logger.info(f"Agent {config.AGENT_NAME} initialized successfully with model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def chat(self, message: str, generate_voice: bool = True) -> Dict[str, Any]:
        """Process a chat message and return response with optional voice synthesis."""
        try:
            if not self.conversation:
                raise RuntimeError("Agent not properly initialized")
            
            # Check cache first if enabled
            if config.PERFORMANCE_SETTINGS.get("enable_caching", True):
                cached_response = self.cache.get(message)
                if cached_response:
                    response_data = {
                        "text": cached_response,
                        "model": self.model_name,
                        "cached": True,
                        "voice_audio": None
                    }
                    
                    # Generate voice if requested and available
                    if generate_voice and self.voice_system:
                        voice_audio = self.voice_system.synthesize_text(cached_response)
                        response_data["voice_audio"] = voice_audio
                    
                    return response_data
            
            # Get response from LLM
            response = self.conversation.predict(input=message)
            
            # Cache the response if enabled
            if config.PERFORMANCE_SETTINGS.get("enable_caching", True):
                self.cache.set(message, response)
            
            # Prepare response data
            response_data = {
                "text": response,
                "model": self.model_name,
                "cached": False,
                "voice_audio": None
            }
            
            # Generate voice if requested and available
            if generate_voice and self.voice_system:
                voice_audio = self.voice_system.synthesize_text(response)
                response_data["voice_audio"] = voice_audio
            
            logger.info(f"Processed message: {message[:50]}...")
            return response_data
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "text": f"I apologize, but I encountered an error: {str(e)}",
                "model": self.model_name,
                "cached": False,
                "voice_audio": None,
                "error": True
            }
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different LLM model."""
        try:
            if self.llm_manager.switch_model(model_name):
                self.model_name = model_name
                logger.info(f"Switched to model: {model_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available LLM models."""
        return self.llm_manager.get_available_models() if self.llm_manager else []
    
    def get_current_model(self) -> str:
        """Get the currently active model."""
        return self.model_name
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models."""
        return self.llm_manager.get_all_performance_metrics() if self.llm_manager else {}
    
    def set_voice(self, voice_name: str) -> bool:
        """Set the voice for speech synthesis."""
        if self.voice_system:
            return self.voice_system.set_voice(voice_name)
        return False
    
    def get_available_voices(self) -> Dict[str, str]:
        """Get available voices for speech synthesis."""
        return self.voice_system.get_available_voices() if self.voice_system else {}
    
    def synthesize_voice(self, text: str, voice_name: Optional[str] = None, 
                        emotion: Optional[str] = None) -> bytes:
        """Synthesize text to speech."""
        if self.voice_system:
            return self.voice_system.synthesize_text(text, voice_name, emotion)
        return b""
    
    def get_voice_info(self) -> Dict[str, Any]:
        """Get information about the voice system."""
        return self.voice_system.get_engine_info() if self.voice_system else {}
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's memory."""
        try:
            if self.memory:
                memory_summary = {
                    "memory_type": type(self.memory).__name__,
                    "memory_size": len(self.memory.memory) if hasattr(self.memory, 'memory') else "Unknown"
                }
            else:
                memory_summary = {"memory_type": "None", "memory_size": 0}
            
            # Add cache statistics
            cache_stats = self.cache.get_stats()
            memory_summary.update({"cache": cache_stats})
            
            # Add model information
            memory_summary.update({
                "current_model": self.model_name,
                "available_models": self.get_available_models(),
                "voice_enabled": self.enable_voice,
                "voice_info": self.get_voice_info()
            })
            
            return memory_summary
        except Exception as e:
            logger.error(f"Error getting memory summary: {e}")
            return {"error": str(e)}
    
    def clear_cache(self):
        """Clear the response cache."""
        self.cache.clear()
        if self.voice_system:
            self.voice_system.clear_cache()
    
    def optimize_memory(self):
        """Optimize memory usage."""
        if self.llm_manager:
            self.llm_manager.optimize_memory()
    
    def export_conversation(self, filepath: str) -> bool:
        """Export conversation history to a file."""
        try:
            if self.memory:
                return self.memory.export_memory(filepath)
            return False
        except Exception as e:
            logger.error(f"Failed to export conversation: {e}")
            return False
    
    def import_conversation(self, filepath: str) -> bool:
        """Import conversation history from a file."""
        try:
            if self.memory:
                return self.memory.import_memory(filepath)
            return False
        except Exception as e:
            logger.error(f"Failed to import conversation: {e}")
            return False

def get_agent(model_name: Optional[str] = None, temperature: Optional[float] = None, 
              enable_voice: bool = True, voice_engine: VoiceEngine = VoiceEngine.TTS) -> AIAgent:
    """Factory function to create an AI agent instance."""
    return AIAgent(
        model_name=model_name or "mistral-7b",
        temperature=temperature or 0.7,
        enable_voice=enable_voice,
        voice_engine=voice_engine
    )

# Backward compatibility
def get_conversation_chain():
    """Legacy function for backward compatibility."""
    agent = get_agent()
    return agent.conversation
