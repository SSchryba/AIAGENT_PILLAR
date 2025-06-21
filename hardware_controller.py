#!/usr/bin/env python3
"""
Hardware Controller for AI Pillar
Manages ESP32, OLED display, RGB ring, and LED strip integration
"""

import asyncio
import json
import logging
import time
import math
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import serial
import serial.tools.list_ports
from dataclasses import dataclass
import threading
import queue

# OLED Display imports
try:
    import board
    import busio
    from PIL import Image, ImageDraw, ImageFont
    import adafruit_ssd1306
    OLED_AVAILABLE = True
except ImportError:
    OLED_AVAILABLE = False
    logging.warning("OLED display libraries not available")

# RGB LED imports
try:
    import neopixel
    RGB_AVAILABLE = True
except ImportError:
    RGB_AVAILABLE = False
    logging.warning("NeoPixel libraries not available")

logger = logging.getLogger(__name__)

class AIState(Enum):
    """AI Agent states for visual feedback"""
    IDLE = "idle"
    THINKING = "thinking"
    SPEAKING = "speaking"
    LISTENING = "listening"
    ERROR = "error"
    STARTUP = "startup"

@dataclass
class HardwareConfig:
    """Hardware configuration for AI Pillar"""
    # ESP32 Configuration
    esp32_port: str = "COM3"  # Windows default, will auto-detect
    esp32_baudrate: int = 115200
    
    # OLED Display Configuration
    oled_width: int = 128
    oled_height: int = 64
    oled_i2c_address: int = 0x3C
    oled_font_size: int = 10
    
    # RGB Ring Configuration
    rgb_ring_pin: int = 18  # GPIO pin for RGB ring
    rgb_ring_count: int = 16  # Number of LEDs in ring
    rgb_ring_brightness: float = 0.3
    
    # LED Strip Configuration
    led_strip_pin: int = 21  # GPIO pin for LED strip
    led_strip_count: int = 60  # Number of LEDs in strip
    led_strip_brightness: float = 0.2
    
    # Animation Settings
    animation_speed: float = 0.1  # seconds between frames
    pulse_speed: float = 0.05  # seconds for pulse animation

class ESP32Controller:
    """Controls ESP32 communication for sensor data and commands"""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.serial_connection = None
        self.is_connected = False
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self._running = False
        self._thread = None
        
    def auto_detect_port(self) -> Optional[str]:
        """Auto-detect ESP32 serial port"""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if "ESP32" in port.description or "CH340" in port.description:
                return port.device
        return None
    
    def connect(self) -> bool:
        """Connect to ESP32"""
        try:
            port = self.config.esp32_port
            if port == "COM3":  # Auto-detect if using default
                detected_port = self.auto_detect_port()
                if detected_port:
                    port = detected_port
            
            self.serial_connection = serial.Serial(
                port=port,
                baudrate=self.config.esp32_baudrate,
                timeout=1
            )
            self.is_connected = True
            self._running = True
            self._thread = threading.Thread(target=self._communication_loop)
            self._thread.daemon = True
            self._thread.start()
            
            logger.info(f"Connected to ESP32 on {port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to ESP32: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from ESP32"""
        self._running = False
        if self.serial_connection:
            self.serial_connection.close()
        self.is_connected = False
        logger.info("Disconnected from ESP32")
    
    def send_command(self, command: str, data: Dict[str, Any] = None) -> bool:
        """Send command to ESP32"""
        if not self.is_connected:
            return False
        
        try:
            message = {
                "command": command,
                "data": data or {},
                "timestamp": time.time()
            }
            self.command_queue.put(message)
            return True
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False
    
    def _communication_loop(self):
        """Background communication loop with ESP32"""
        while self._running and self.is_connected:
            try:
                # Send commands
                if not self.command_queue.empty():
                    message = self.command_queue.get()
                    json_message = json.dumps(message) + "\n"
                    self.serial_connection.write(json_message.encode())
                
                # Read responses
                if self.serial_connection.in_waiting:
                    response = self.serial_connection.readline().decode().strip()
                    if response:
                        try:
                            data = json.loads(response)
                            self.response_queue.put(data)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON from ESP32: {response}")
                
                time.sleep(0.01)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"ESP32 communication error: {e}")
                self.is_connected = False
                break

class OLEDController:
    """Controls 0.96" OLED display for visual feedback"""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.display = None
        self.font = None
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize OLED display"""
        if not OLED_AVAILABLE:
            logger.warning("OLED libraries not available")
            return False
        
        try:
            # Initialize I2C bus
            i2c = busio.I2C(board.SCL, board.SDA)
            
            # Initialize OLED display
            self.display = adafruit_ssd1306.SSD1306_I2C(
                self.config.oled_width,
                self.config.oled_height,
                i2c,
                addr=self.config.oled_i2c_address
            )
            
            # Clear display
            self.display.fill(0)
            self.display.show()
            
            # Create font (using default system font)
            self.font = ImageFont.load_default()
            
            self.is_initialized = True
            logger.info("OLED display initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OLED display: {e}")
            return False
    
    def clear(self):
        """Clear the display"""
        if self.is_initialized:
            self.display.fill(0)
            self.display.show()
    
    def show_text(self, text: str, x: int = 0, y: int = 0, clear: bool = True):
        """Display text on OLED"""
        if not self.is_initialized:
            return
        
        try:
            if clear:
                self.display.fill(0)
            
            # Create image and draw object
            image = Image.new("1", (self.config.oled_width, self.config.oled_height))
            draw = ImageDraw.Draw(image)
            
            # Draw text
            draw.text((x, y), text, font=self.font, fill=255)
            
            # Display image
            self.display.image(image)
            self.display.show()
            
        except Exception as e:
            logger.error(f"Failed to display text on OLED: {e}")
    
    def show_synth_wave(self, frequency: float = 1.0, amplitude: float = 1.0):
        """Display synthetic wave animation for AI speaking"""
        if not self.is_initialized:
            return
        
        try:
            self.display.fill(0)
            
            # Create wave pattern
            for x in range(self.config.oled_width):
                y = int(self.config.oled_height / 2 + 
                       amplitude * 10 * 
                       math.sin(2 * math.pi * frequency * x / self.config.oled_width))
                if 0 <= y < self.config.oled_height:
                    self.display.pixel(x, y, 1)
            
            self.display.show()
            
        except Exception as e:
            logger.error(f"Failed to display synth wave: {e}")

class RGBRingController:
    """Controls RGB LED ring for AI state indication"""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.pixels = None
        self.is_initialized = False
        self._animation_thread = None
        self._running = False
        
    def initialize(self) -> bool:
        """Initialize RGB ring"""
        if not RGB_AVAILABLE:
            logger.warning("RGB libraries not available")
            return False
        
        try:
            self.pixels = neopixel.NeoPixel(
                board.D18,  # GPIO pin
                self.config.rgb_ring_count,
                brightness=self.config.rgb_ring_brightness,
                auto_write=False
            )
            
            # Clear all pixels
            self.pixels.fill((0, 0, 0))
            self.pixels.show()
            
            self.is_initialized = True
            logger.info("RGB ring initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RGB ring: {e}")
            return False
    
    def set_color(self, color: Tuple[int, int, int]):
        """Set all LEDs to a specific color"""
        if self.is_initialized:
            self.pixels.fill(color)
            self.pixels.show()
    
    def set_pixel(self, pixel: int, color: Tuple[int, int, int]):
        """Set specific pixel to color"""
        if self.is_initialized and 0 <= pixel < self.config.rgb_ring_count:
            self.pixels[pixel] = color
            self.pixels.show()
    
    def pulse_animation(self, color: Tuple[int, int, int], duration: float = 2.0):
        """Pulse animation for AI awake state"""
        if not self.is_initialized:
            return
        
        self._running = True
        self._animation_thread = threading.Thread(
            target=self._pulse_loop, 
            args=(color, duration)
        )
        self._animation_thread.daemon = True
        self._animation_thread.start()
    
    def _pulse_loop(self, color: Tuple[int, int, int], duration: float):
        """Pulse animation loop"""
        start_time = time.time()
        while self._running and (time.time() - start_time) < duration:
            # Fade in
            for brightness in range(0, 101, 5):
                if not self._running:
                    break
                adjusted_color = tuple(int(c * brightness / 100) for c in color)
                self.pixels.fill(adjusted_color)
                self.pixels.show()
                time.sleep(self.config.pulse_speed)
            
            # Fade out
            for brightness in range(100, -1, -5):
                if not self._running:
                    break
                adjusted_color = tuple(int(c * brightness / 100) for c in color)
                self.pixels.fill(adjusted_color)
                self.pixels.show()
                time.sleep(self.config.pulse_speed)
    
    def stop_animation(self):
        """Stop current animation"""
        self._running = False
        if self._animation_thread:
            self._animation_thread.join(timeout=1.0)

class LEDStripController:
    """Controls LED strip for thinking/processing indication"""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.pixels = None
        self.is_initialized = False
        self._animation_thread = None
        self._running = False
        
    def initialize(self) -> bool:
        """Initialize LED strip"""
        if not RGB_AVAILABLE:
            logger.warning("RGB libraries not available")
            return False
        
        try:
            self.pixels = neopixel.NeoPixel(
                board.D21,  # GPIO pin
                self.config.led_strip_count,
                brightness=self.config.led_strip_brightness,
                auto_write=False
            )
            
            # Clear all pixels
            self.pixels.fill((0, 0, 0))
            self.pixels.show()
            
            self.is_initialized = True
            logger.info("LED strip initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LED strip: {e}")
            return False
    
    def thinking_animation(self, color: Tuple[int, int, int] = (0, 0, 255)):
        """Top-to-bottom thinking animation"""
        if not self.is_initialized:
            return
        
        self._running = True
        self._animation_thread = threading.Thread(
            target=self._thinking_loop, 
            args=(color,)
        )
        self._animation_thread.daemon = True
        self._animation_thread.start()
    
    def _thinking_loop(self, color: Tuple[int, int, int]):
        """Thinking animation loop - top to bottom pulse"""
        while self._running:
            # Clear strip
            self.pixels.fill((0, 0, 0))
            
            # Create wave from top to bottom
            for i in range(self.config.led_strip_count):
                if not self._running:
                    break
                
                # Calculate brightness based on position
                brightness = int(255 * (1 - abs(i - self.config.led_strip_count / 2) / (self.config.led_strip_count / 2)))
                adjusted_color = tuple(int(c * brightness / 255) for c in color)
                
                self.pixels[i] = adjusted_color
                self.pixels.show()
                time.sleep(self.config.animation_speed)
    
    def stop_animation(self):
        """Stop current animation"""
        self._running = False
        if self._animation_thread:
            self._animation_thread.join(timeout=1.0)

class HardwareController:
    """Main hardware controller for AI Pillar"""
    
    def __init__(self, config: HardwareConfig = None):
        self.config = config or HardwareConfig()
        self.esp32 = ESP32Controller(self.config)
        self.oled = OLEDController(self.config)
        self.rgb_ring = RGBRingController(self.config)
        self.led_strip = LEDStripController(self.config)
        self.current_state = AIState.IDLE
        
    def initialize(self) -> bool:
        """Initialize all hardware components"""
        try:
            logger.info("Initializing AI Pillar hardware...")
            
            # Initialize ESP32
            esp32_ok = self.esp32.connect()
            
            # Initialize OLED display
            oled_ok = self.oled.initialize()
            
            # Initialize RGB ring
            rgb_ok = self.rgb_ring.initialize()
            
            # Initialize LED strip
            strip_ok = self.led_strip.initialize()
            
            if esp32_ok or oled_ok or rgb_ok or strip_ok:
                logger.info("Hardware initialization completed")
                return True
            else:
                logger.error("No hardware components could be initialized")
                return False
                
        except Exception as e:
            logger.error(f"Hardware initialization failed: {e}")
            return False
    
    def set_ai_state(self, state: AIState):
        """Set AI state and update visual feedback"""
        self.current_state = state
        
        try:
            if state == AIState.IDLE:
                self._set_idle_state()
            elif state == AIState.THINKING:
                self._set_thinking_state()
            elif state == AIState.SPEAKING:
                self._set_speaking_state()
            elif state == AIState.LISTENING:
                self._set_listening_state()
            elif state == AIState.ERROR:
                self._set_error_state()
            elif state == AIState.STARTUP:
                self._set_startup_state()
                
        except Exception as e:
            logger.error(f"Failed to set AI state {state}: {e}")
    
    def _set_idle_state(self):
        """Set idle state - minimal lighting"""
        self.rgb_ring.stop_animation()
        self.led_strip.stop_animation()
        self.rgb_ring.set_color((0, 0, 10))  # Dim blue
        self.led_strip.pixels.fill((0, 0, 0))
        self.led_strip.pixels.show()
        self.oled.show_text("AI Ready")
    
    def _set_thinking_state(self):
        """Set thinking state - LED strip animation"""
        self.rgb_ring.stop_animation()
        self.led_strip.thinking_animation((0, 0, 255))  # Blue thinking
        self.oled.show_text("Thinking...")
    
    def _set_speaking_state(self):
        """Set speaking state - RGB ring pulse + OLED wave"""
        self.led_strip.stop_animation()
        self.rgb_ring.pulse_animation((0, 255, 0), duration=0)  # Green pulse
        self.oled.show_synth_wave(frequency=2.0, amplitude=0.8)
    
    def _set_listening_state(self):
        """Set listening state - RGB ring pulse"""
        self.led_strip.stop_animation()
        self.rgb_ring.pulse_animation((255, 255, 0), duration=0)  # Yellow pulse
        self.oled.show_text("Listening...")
    
    def _set_error_state(self):
        """Set error state - red indication"""
        self.rgb_ring.stop_animation()
        self.led_strip.stop_animation()
        self.rgb_ring.set_color((255, 0, 0))  # Red
        self.oled.show_text("Error")
    
    def _set_startup_state(self):
        """Set startup state - rainbow animation"""
        self.rgb_ring.stop_animation()
        self.led_strip.stop_animation()
        # Startup animation would go here
        self.oled.show_text("Starting...")
    
    def cleanup(self):
        """Cleanup hardware resources"""
        try:
            self.rgb_ring.stop_animation()
            self.led_strip.stop_animation()
            self.esp32.disconnect()
            
            # Turn off all LEDs
            if self.rgb_ring.is_initialized:
                self.rgb_ring.set_color((0, 0, 0))
            if self.led_strip.is_initialized:
                self.led_strip.pixels.fill((0, 0, 0))
                self.led_strip.pixels.show()
            
            logger.info("Hardware cleanup completed")
            
        except Exception as e:
            logger.error(f"Hardware cleanup failed: {e}")

# Global hardware controller instance
_hardware_controller = None

def get_hardware_controller(config: HardwareConfig = None) -> HardwareController:
    """Get or create hardware controller instance"""
    global _hardware_controller
    if _hardware_controller is None:
        _hardware_controller = HardwareController(config)
    return _hardware_controller

def initialize_hardware(config: HardwareConfig = None) -> bool:
    """Initialize hardware controller"""
    controller = get_hardware_controller(config)
    return controller.initialize()

def set_ai_state(state: AIState):
    """Set AI state for visual feedback"""
    controller = get_hardware_controller()
    controller.set_ai_state(state)

def cleanup_hardware():
    """Cleanup hardware resources"""
    global _hardware_controller
    if _hardware_controller:
        _hardware_controller.cleanup()
        _hardware_controller = None 