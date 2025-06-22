#!/usr/bin/env python3
"""
AI Pillar Integration Module - Raspberry Pi 4 Direct Control
Connects AI Agent to hardware components (OLED, RGB LEDs, Touchscreen)
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

from agent import get_agent, AIAgent
from hardware_controller import (
    get_hardware_controller, 
    HardwareController, 
    HardwareConfig, 
    AIState
)
from voice_system import get_voice_system, VoiceConfig, VoiceEngine
import agent_config as config

logger = logging.getLogger(__name__)

class PillarMode(Enum):
    """AI Pillar operating modes"""
    STANDALONE = "standalone"  # Run on Raspberry Pi with direct hardware control
    SIMULATION = "simulation"  # Run without hardware for testing

@dataclass
class PillarConfig:
    """Configuration for AI Pillar integration"""
    mode: PillarMode = PillarMode.STANDALONE
    enable_voice: bool = True
    enable_visual_feedback: bool = True
    enable_touchscreen: bool = True
    auto_initialize: bool = True

class AIPillarIntegration:
    """Main integration class for AI Pillar on Raspberry Pi 4"""
    
    def __init__(self, pillar_config: PillarConfig = None):
        self.config = pillar_config or PillarConfig()
        self.agent = None
        self.hardware_controller = None
        self.voice_system = None
        self.is_running = False
        
    async def initialize(self) -> bool:
        """Initialize AI Pillar integration"""
        try:
            logger.info("Initializing Raspberry Pi 4 AI Pillar integration...")
            
            # Initialize AI Agent
            self.agent = get_agent(
                model_name=config.MODEL_CONFIG["default_model"],
                enable_voice=self.config.enable_voice
            )
            
            # Initialize voice system if enabled
            if self.config.enable_voice:
                voice_config = VoiceConfig(
                    engine=VoiceEngine.TTS,
                    voice_name=config.VOICE_SETTINGS["voice_name"],
                    quality=config.VOICE_SETTINGS["quality"],
                    speed=config.VOICE_SETTINGS["speed"],
                    volume=config.VOICE_SETTINGS["volume"],
                    language=config.VOICE_SETTINGS["language"]
                )
                self.voice_system = get_voice_system(voice_config)
            
            # Initialize hardware controller
            if self.config.enable_visual_feedback:
                hw_config = HardwareConfig()
                self.hardware_controller = get_hardware_controller(hw_config)
                if not self.hardware_controller.initialize():
                    logger.warning("Hardware initialization failed - running in simulation mode")
            
            # Set initial state
            await self.set_ai_state(AIState.STARTUP)
            
            logger.info("AI Pillar integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Pillar: {e}")
            return False
    
    async def set_ai_state(self, state: AIState):
        """Set AI state and update visual feedback"""
        if self.hardware_controller:
            self.hardware_controller.set_ai_state(state)
        
        logger.info(f"AI State changed to: {state.value}")
    
    async def process_message(self, message: str) -> Dict[str, Any]:
        """Process a message through the AI agent with visual feedback"""
        try:
            # Set thinking state
            await self.set_ai_state(AIState.THINKING)
            
            # Process with AI agent
            response = await self.agent.process_message(message)
            
            # Set speaking state if voice is enabled
            if self.config.enable_voice and self.voice_system:
                await self.set_ai_state(AIState.SPEAKING)
                
                # Speak the response
                if response.get("response"):
                    await self.voice_system.speak(response["response"])
            
            # Return to idle state
            await self.set_ai_state(AIState.IDLE)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.set_ai_state(AIState.ERROR)
            
            return {
                "success": False,
                "error": str(e),
                "response": "I encountered an error processing your request."
            }
    
    async def start_listening(self):
        """Start voice listening mode"""
        await self.set_ai_state(AIState.LISTENING)
        # Voice listening implementation would go here
    
    async def stop_listening(self):
        """Stop voice listening mode"""
        await self.set_ai_state(AIState.IDLE)
    
    async def shutdown(self):
        """Shutdown AI Pillar integration"""
        try:
            logger.info("Shutting down AI Pillar integration...")
            
            # Set shutdown state
            await self.set_ai_state(AIState.ERROR)
            
            # Clean up hardware
            if self.hardware_controller:
                self.hardware_controller.cleanup()
            
            # Clean up voice system
            if self.voice_system:
                # Voice system cleanup would go here
                pass
            
            self.is_running = False
            logger.info("AI Pillar integration shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

class AIPillarWebInterface:
    """Web interface for AI Pillar with touchscreen support"""
    
    def __init__(self, pillar_integration: AIPillarIntegration):
        self.pillar = pillar_integration
    
    async def handle_chat_request(self, message: str) -> Dict[str, Any]:
        """Handle chat request from web interface"""
        return await self.pillar.process_message(message)
    
    async def get_hardware_status(self) -> Dict[str, Any]:
        """Get hardware status for web interface"""
        if not self.pillar.hardware_controller:
            return {"status": "no_hardware"}
        
        return {
            "status": "connected",
            "mode": self.pillar.config.mode.value,
            "current_state": self.pillar.hardware_controller.current_state.value,
            "components": {
                "oled": self.pillar.hardware_controller.oled.is_initialized,
                "rgb_ring": self.pillar.hardware_controller.rgb_ring.is_initialized,
                "led_strip": self.pillar.hardware_controller.led_strip.is_initialized
            }
        }
    
    async def set_ai_state(self, state: str) -> Dict[str, Any]:
        """Set AI state via web interface"""
        try:
            ai_state = AIState(state)
            await self.pillar.set_ai_state(ai_state)
            return {"success": True, "state": state}
        except ValueError:
            return {"success": False, "error": f"Invalid state: {state}"}

# Global integration instance
_global_pillar = None

async def get_pillar_integration(config: PillarConfig = None) -> AIPillarIntegration:
    """Get the global AI Pillar integration instance"""
    global _global_pillar
    if _global_pillar is None:
        _global_pillar = AIPillarIntegration(config)
    return _global_pillar

async def get_pillar_web_interface(config: PillarConfig = None) -> AIPillarWebInterface:
    """Get web interface for AI Pillar"""
    pillar = await get_pillar_integration(config)
    return AIPillarWebInterface(pillar)

async def initialize_pillar(config: PillarConfig = None) -> bool:
    """Initialize AI Pillar integration"""
    pillar = await get_pillar_integration(config)
    return await pillar.initialize()

async def shutdown_pillar():
    """Shutdown AI Pillar integration"""
    global _global_pillar
    if _global_pillar:
        await _global_pillar.shutdown()
        _global_pillar = None

async def example_chat_session():
    """Example chat session with visual feedback"""
    try:
        # Initialize pillar
        config = PillarConfig(
            mode=PillarMode.STANDALONE,
            enable_voice=True,
            enable_visual_feedback=True
        )
        
        success = await initialize_pillar(config)
        if not success:
            logger.error("Failed to initialize AI Pillar")
            return
        
        # Example conversation
        messages = [
            "Hello! How are you today?",
            "What's the weather like?",
            "Tell me a joke",
            "Goodbye!"
        ]
        
        for message in messages:
            logger.info(f"User: {message}")
            response = await process_message(message)
            logger.info(f"AI: {response.get('response', 'No response')}")
            await asyncio.sleep(2)  # Pause between messages
        
        # Shutdown
        await shutdown_pillar()
        
    except Exception as e:
        logger.error(f"Example session error: {e}")

# Convenience function for processing messages
async def process_message(message: str) -> Dict[str, Any]:
    """Process a message through the AI Pillar"""
    pillar = await get_pillar_integration()
    return await pillar.process_message(message)

if __name__ == "__main__":
    # Run example
    asyncio.run(example_chat_session()) 