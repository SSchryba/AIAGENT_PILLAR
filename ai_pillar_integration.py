#!/usr/bin/env python3
"""
AI Pillar Integration Module
Connects AI Agent to hardware components (ESP32, OLED, RGB LEDs)
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional
from enum import Enum
import websockets
import serial
import threading
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
    NETWORK = "network"        # Run on any device, communicate with ESP32 via network
    HYBRID = "hybrid"          # Use both local hardware and ESP32

@dataclass
class PillarConfig:
    """Configuration for AI Pillar integration"""
    mode: PillarMode = PillarMode.NETWORK
    esp32_ip: str = "192.168.1.100"  # ESP32 IP address
    esp32_port: int = 81  # ESP32 WebSocket port
    esp32_serial_port: str = "COM3"  # ESP32 serial port (for direct connection)
    enable_voice: bool = True
    enable_visual_feedback: bool = True
    auto_connect: bool = True

class ESP32Communicator:
    """Handles communication with ESP32 via WebSocket"""
    
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.websocket = None
        self.is_connected = False
        self._running = False
        self._thread = None
        
    async def connect(self) -> bool:
        """Connect to ESP32 WebSocket server"""
        try:
            uri = f"ws://{self.ip}:{self.port}"
            self.websocket = await websockets.connect(uri)
            self.is_connected = True
            self._running = True
            
            # Start message handling thread
            self._thread = threading.Thread(target=self._message_loop)
            self._thread.daemon = True
            self._thread.start()
            
            logger.info(f"Connected to ESP32 at {uri}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to ESP32: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from ESP32"""
        self._running = False
        if self.websocket:
            asyncio.create_task(self.websocket.close())
        self.is_connected = False
        logger.info("Disconnected from ESP32")
    
    async def send_command(self, command: str, data: Dict[str, Any] = None) -> bool:
        """Send command to ESP32"""
        if not self.is_connected or not self.websocket:
            return False
        
        try:
            message = {
                "command": command,
                "data": data or {},
                "timestamp": time.time()
            }
            await self.websocket.send(json.dumps(message))
            return True
            
        except Exception as e:
            logger.error(f"Failed to send command to ESP32: {e}")
            return False
    
    def _message_loop(self):
        """Background message handling loop"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def _async_loop():
            while self._running and self.is_connected:
                try:
                    if self.websocket:
                        message = await self.websocket.recv()
                        self._handle_message(message)
                except Exception as e:
                    logger.error(f"ESP32 message loop error: {e}")
                    break
        
        loop.run_until_complete(_async_loop())
        loop.close()
    
    def _handle_message(self, message: str):
        """Handle incoming messages from ESP32"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "status":
                logger.info(f"ESP32 Status: {data.get('state')}")
            elif msg_type == "state_change":
                logger.info(f"ESP32 State Change: {data.get('state')}")
            elif msg_type == "sensor_data":
                # Handle sensor data if needed
                pass
                
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from ESP32: {message}")

class AIPillarIntegration:
    """Main integration class for AI Pillar"""
    
    def __init__(self, pillar_config: PillarConfig = None):
        self.config = pillar_config or PillarConfig()
        self.agent = None
        self.hardware_controller = None
        self.esp32_communicator = None
        self.voice_system = None
        self.is_running = False
        
    async def initialize(self) -> bool:
        """Initialize AI Pillar integration"""
        try:
            logger.info("Initializing AI Pillar integration...")
            
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
            
            # Initialize hardware based on mode
            if self.config.mode in [PillarMode.STANDALONE, PillarMode.HYBRID]:
                # Initialize local hardware controller
                hw_config = HardwareConfig(
                    esp32_port=self.config.esp32_serial_port
                )
                self.hardware_controller = get_hardware_controller(hw_config)
                if not self.hardware_controller.initialize():
                    logger.warning("Failed to initialize local hardware")
            
            if self.config.mode in [PillarMode.NETWORK, PillarMode.HYBRID]:
                # Initialize ESP32 communicator
                self.esp32_communicator = ESP32Communicator(
                    self.config.esp32_ip,
                    self.config.esp32_port
                )
                if self.config.auto_connect:
                    await self.esp32_communicator.connect()
            
            # Set initial state
            await self.set_ai_state(AIState.STARTUP)
            
            logger.info("AI Pillar integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Pillar: {e}")
            return False
    
    async def set_ai_state(self, state: AIState):
        """Set AI state and update all hardware components"""
        try:
            # Update local hardware if available
            if self.hardware_controller:
                self.hardware_controller.set_ai_state(state)
            
            # Update ESP32 if connected
            if self.esp32_communicator and self.esp32_communicator.is_connected:
                await self.esp32_communicator.send_command("set_state", {
                    "state": state.value
                })
            
            logger.info(f"AI State set to: {state.value}")
            
        except Exception as e:
            logger.error(f"Failed to set AI state: {e}")
    
    async def process_message(self, message: str) -> Dict[str, Any]:
        """Process a message through the AI Agent with visual feedback"""
        try:
            # Set thinking state
            await self.set_ai_state(AIState.THINKING)
            
            # Process with AI Agent
            response_data = self.agent.chat(message, generate_voice=self.config.enable_voice)
            
            # Set speaking state if voice is enabled
            if self.config.enable_voice and response_data.get("voice_audio"):
                await self.set_ai_state(AIState.SPEAKING)
                
                # Play voice response
                if self.voice_system:
                    # This would integrate with your audio system
                    logger.info("Voice response generated")
            
            # Return to idle state
            await self.set_ai_state(AIState.IDLE)
            
            return response_data
            
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            await self.set_ai_state(AIState.ERROR)
            return {"error": str(e)}
    
    async def start_listening(self):
        """Start listening mode"""
        await self.set_ai_state(AIState.LISTENING)
    
    async def stop_listening(self):
        """Stop listening mode"""
        await self.set_ai_state(AIState.IDLE)
    
    async def shutdown(self):
        """Shutdown AI Pillar integration"""
        try:
            logger.info("Shutting down AI Pillar...")
            
            # Set shutdown state
            await self.set_ai_state(AIState.ERROR)
            
            # Cleanup hardware
            if self.hardware_controller:
                self.hardware_controller.cleanup()
            
            # Disconnect from ESP32
            if self.esp32_communicator:
                self.esp32_communicator.disconnect()
            
            self.is_running = False
            logger.info("AI Pillar shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

class AIPillarWebInterface:
    """Web interface specifically for AI Pillar"""
    
    def __init__(self, pillar_integration: AIPillarIntegration):
        self.pillar = pillar_integration
        
    async def handle_chat_request(self, message: str) -> Dict[str, Any]:
        """Handle chat request with visual feedback"""
        return await self.pillar.process_message(message)
    
    async def get_hardware_status(self) -> Dict[str, Any]:
        """Get hardware status"""
        status = {
            "mode": self.pillar.config.mode.value,
            "hardware_available": self.pillar.hardware_controller is not None,
            "esp32_connected": (
                self.pillar.esp32_communicator.is_connected 
                if self.pillar.esp32_communicator else False
            ),
            "voice_enabled": self.pillar.config.enable_voice,
            "visual_feedback_enabled": self.pillar.config.enable_visual_feedback
        }
        return status

# Global instances
_pillar_integration = None
_pillar_web_interface = None

async def get_pillar_integration(config: PillarConfig = None) -> AIPillarIntegration:
    """Get or create AI Pillar integration instance"""
    global _pillar_integration
    if _pillar_integration is None:
        _pillar_integration = AIPillarIntegration(config)
        await _pillar_integration.initialize()
    return _pillar_integration

async def get_pillar_web_interface(config: PillarConfig = None) -> AIPillarWebInterface:
    """Get or create AI Pillar web interface instance"""
    global _pillar_web_interface
    if _pillar_web_interface is None:
        pillar = await get_pillar_integration(config)
        _pillar_web_interface = AIPillarWebInterface(pillar)
    return _pillar_web_interface

async def initialize_pillar(config: PillarConfig = None) -> bool:
    """Initialize AI Pillar with given configuration"""
    pillar = await get_pillar_integration(config)
    return pillar is not None

async def shutdown_pillar():
    """Shutdown AI Pillar"""
    global _pillar_integration, _pillar_web_interface
    if _pillar_integration:
        await _pillar_integration.shutdown()
        _pillar_integration = None
    _pillar_web_interface = None

# Example usage functions
async def example_chat_session():
    """Example of how to use the AI Pillar integration"""
    # Initialize with network mode (communicate with ESP32)
    config = PillarConfig(
        mode=PillarMode.NETWORK,
        esp32_ip="192.168.1.100",
        enable_voice=True,
        enable_visual_feedback=True
    )
    
    pillar = await get_pillar_integration(config)
    
    # Process a message
    response = await pillar.process_message("Hello, how are you today?")
    print(f"AI Response: {response['text']}")
    
    # Shutdown
    await shutdown_pillar()

if __name__ == "__main__":
    # Run example
    asyncio.run(example_chat_session()) 