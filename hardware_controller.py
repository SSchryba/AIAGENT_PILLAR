#!/usr/bin/env python3
"""
Hardware Controller for AI Pillar - Raspberry Pi 4 Direct Control
Manages OLED display, RGB ring, and LED strip directly via GPIO
"""

import asyncio
import json
import logging
import time
import math
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass
import threading
import queue

# Raspberry Pi GPIO imports
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logging.warning("RPi.GPIO not available - running in simulation mode")

# OLED Display imports for Raspberry Pi
try:
    import board
    import busio
    from PIL import Image, ImageDraw, ImageFont
    import adafruit_ssd1306
    OLED_AVAILABLE = True
except ImportError:
    OLED_AVAILABLE = False
    logging.warning("OLED display libraries not available")

# RGB LED imports for Raspberry Pi
try:
    import neopixel
    RGB_AVAILABLE = True
except ImportError:
    RGB_AVAILABLE = False
    logging.warning("NeoPixel libraries not available")

# Pi Screen imports for touchscreen
try:
    import pygame
    import random
    import math
    PI_SCREEN_AVAILABLE = True
except ImportError:
    PI_SCREEN_AVAILABLE = False
    logging.warning("Pygame not available - Pi screen functionality disabled")

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
    """Hardware configuration for Raspberry Pi 4 AI Pillar"""
    # OLED Display Configuration
    oled_width: int = 128
    oled_height: int = 64
    oled_i2c_address: int = 0x3C
    oled_font_size: int = 10
    
    # RGB Ring Configuration (WS2812B)
    rgb_ring_pin: int = 18  # GPIO pin for RGB ring
    rgb_ring_count: int = 16  # Number of LEDs in ring
    rgb_ring_brightness: float = 0.3
    
    # LED Strip Configuration (WS2812B)
    led_strip_pin: int = 21  # GPIO pin for LED strip
    led_strip_count: int = 60  # Number of LEDs in strip
    led_strip_brightness: float = 0.2
    
    # Animation Settings
    animation_speed: float = 0.1  # seconds between frames
    pulse_speed: float = 0.05  # seconds for pulse animation
    
    # GPIO Configuration
    gpio_mode: str = "BCM"  # Use BCM pin numbering

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
            logger.warning("OLED libraries not available - running in simulation mode")
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
            
            # Create font
            self.font = ImageFont.load_default()
            
            # Clear display
            self.display.fill(0)
            self.display.show()
            
            self.is_initialized = True
            logger.info("OLED display initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OLED display: {e}")
            return False
    
    def clear(self):
        """Clear the OLED display"""
        if self.is_initialized and self.display:
            self.display.fill(0)
            self.display.show()
    
    def show_text(self, text: str, x: int = 0, y: int = 0, clear: bool = True):
        """Display text on OLED"""
        if not self.is_initialized or not self.display:
            logger.info(f"OLED Text (simulation): {text}")
            return
        
        try:
            if clear:
                self.display.fill(0)
            
            # Create image with text
            image = Image.new("1", (self.config.oled_width, self.config.oled_height))
            draw = ImageDraw.Draw(image)
            draw.text((x, y), text, font=self.font, fill=255)
            
            # Display image
            self.display.image(image)
            self.display.show()
            
        except Exception as e:
            logger.error(f"Failed to display text on OLED: {e}")
    
    def show_synth_wave(self, frequency: float = 1.0, amplitude: float = 1.0):
        """Display synth wave animation on OLED"""
        if not self.is_initialized or not self.display:
            logger.info("OLED Synth Wave (simulation)")
            return
        
        try:
            # Create wave pattern
            image = Image.new("1", (self.config.oled_width, self.config.oled_height))
            draw = ImageDraw.Draw(image)
            
            # Draw sine wave
            for x in range(self.config.oled_width):
                y = int(self.config.oled_height / 2 + 
                       amplitude * 20 * math.sin(frequency * x * 0.1 + time.time() * 2))
                if 0 <= y < self.config.oled_height:
                    draw.point((x, y), fill=255)
            
            self.display.image(image)
            self.display.show()
            
        except Exception as e:
            logger.error(f"Failed to display synth wave: {e}")

class RGBRingController:
    """Controls RGB LED ring using WS2812B protocol"""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.pixels = None
        self.is_initialized = False
        self._animation_thread = None
        self._stop_animation = False
        
    def initialize(self) -> bool:
        """Initialize RGB ring"""
        if not RGB_AVAILABLE or not GPIO_AVAILABLE:
            logger.warning("RGB libraries not available - running in simulation mode")
            return False
        
        try:
            # Set up GPIO mode
            GPIO.setmode(GPIO.BCM)
            
            # Initialize NeoPixel strip
            self.pixels = neopixel.NeoPixel(
                self.config.rgb_ring_pin,
                self.config.rgb_ring_count,
                brightness=self.config.rgb_ring_brightness,
                auto_write=False,
                pixel_order=neopixel.GRB
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
        """Set all LEDs to the same color"""
        if not self.is_initialized or not self.pixels:
            logger.info(f"RGB Ring Color (simulation): {color}")
            return
        
        try:
            self.pixels.fill(color)
            self.pixels.show()
        except Exception as e:
            logger.error(f"Failed to set RGB ring color: {e}")
    
    def set_pixel(self, pixel: int, color: Tuple[int, int, int]):
        """Set a specific pixel color"""
        if not self.is_initialized or not self.pixels:
            return
        
        try:
            if 0 <= pixel < self.config.rgb_ring_count:
                self.pixels[pixel] = color
                self.pixels.show()
        except Exception as e:
            logger.error(f"Failed to set pixel {pixel}: {e}")
    
    def pulse_animation(self, color: Tuple[int, int, int], duration: float = 2.0):
        """Start pulse animation"""
        if self._animation_thread and self._animation_thread.is_alive():
            self.stop_animation()
        
        self._stop_animation = False
        self._animation_thread = threading.Thread(
            target=self._pulse_loop,
            args=(color, duration)
        )
        self._animation_thread.daemon = True
        self._animation_thread.start()
    
    def _pulse_loop(self, color: Tuple[int, int, int], duration: float):
        """Pulse animation loop"""
        start_time = time.time()
        
        while not self._stop_animation and (time.time() - start_time) < duration:
            try:
                # Calculate brightness based on sine wave
                brightness = abs(math.sin(time.time() * 3)) * 0.8 + 0.2
                
                # Apply brightness to color
                dimmed_color = tuple(int(c * brightness) for c in color)
                
                if self.is_initialized and self.pixels:
                    self.pixels.fill(dimmed_color)
                    self.pixels.show()
                
                time.sleep(self.config.pulse_speed)
                
            except Exception as e:
                logger.error(f"Pulse animation error: {e}")
                break
        
        # Turn off LEDs when animation ends
        if self.is_initialized and self.pixels:
            self.pixels.fill((0, 0, 0))
            self.pixels.show()
    
    def stop_animation(self):
        """Stop current animation"""
        self._stop_animation = True
        if self._animation_thread and self._animation_thread.is_alive():
            self._animation_thread.join(timeout=1.0)

class LEDStripController:
    """Controls LED strip using WS2812B protocol"""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.pixels = None
        self.is_initialized = False
        self._animation_thread = None
        self._stop_animation = False
        
    def initialize(self) -> bool:
        """Initialize LED strip"""
        if not RGB_AVAILABLE or not GPIO_AVAILABLE:
            logger.warning("LED strip libraries not available - running in simulation mode")
            return False
        
        try:
            # Set up GPIO mode
            GPIO.setmode(GPIO.BCM)
            
            # Initialize NeoPixel strip
            self.pixels = neopixel.NeoPixel(
                self.config.led_strip_pin,
                self.config.led_strip_count,
                brightness=self.config.led_strip_brightness,
                auto_write=False,
                pixel_order=neopixel.GRB
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
        """Start thinking wave animation"""
        if self._animation_thread and self._animation_thread.is_alive():
            self.stop_animation()
        
        self._stop_animation = False
        self._animation_thread = threading.Thread(
            target=self._thinking_loop,
            args=(color,)
        )
        self._animation_thread.daemon = True
        self._animation_thread.start()
    
    def _thinking_loop(self, color: Tuple[int, int, int]):
        """Thinking animation loop - wave from top to bottom"""
        while not self._stop_animation:
            try:
                for i in range(self.config.led_strip_count):
                    if self._stop_animation:
                        break
                    
                    # Calculate wave position
                    wave_pos = (time.time() * 2 + i * 0.2) % (self.config.led_strip_count * 2)
                    brightness = abs(math.sin(wave_pos)) * 0.6 + 0.2
                    
                    # Apply brightness to color
                    dimmed_color = tuple(int(c * brightness) for c in color)
                    
                    if self.is_initialized and self.pixels:
                        self.pixels[i] = dimmed_color
                        self.pixels.show()
                    
                    time.sleep(self.config.animation_speed)
                
            except Exception as e:
                logger.error(f"Thinking animation error: {e}")
                break
        
        # Turn off LEDs when animation ends
        if self.is_initialized and self.pixels:
            self.pixels.fill((0, 0, 0))
            self.pixels.show()
    
    def stop_animation(self):
        """Stop current animation"""
        self._stop_animation = True
        if self._animation_thread and self._animation_thread.is_alive():
            self._animation_thread.join(timeout=1.0)

class PiScreenController:
    """Controls Raspberry Pi 4 touchscreen for standby visual"""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.screen = None
        self.clock = None
        self.is_initialized = False
        self._animation_thread = None
        self._stop_animation = False
        
        # Screen setup for Pi 4 touchscreen (800x480 resolution)
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 480
        self.FPS = 60
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.FACE_COLOR = (120, 180, 255)
        self.EYE_COLOR_BASE = (0, 255, 255)
        
        # Animation state
        self.particles = []
        self.face_angle = 0
        self.eye_blink_timer = 0
        self.eye_blink_duration = 0
        
    def initialize(self) -> bool:
        """Initialize Pi touchscreen"""
        if not PI_SCREEN_AVAILABLE:
            logger.warning("Pygame not available - Pi screen functionality disabled")
            return False
        
        try:
            # Initialize pygame
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("AI Pillar - Mystical Floating Face")
            self.clock = pygame.time.Clock()
            
            # Initialize particles
            self._init_particles()
            
            self.is_initialized = True
            logger.info("Pi touchscreen initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Pi touchscreen: {e}")
            return False
    
    def _init_particles(self):
        """Initialize background mist particles"""
        self.particles = []
        for _ in range(50):  # Create 50 particles
            self.particles.append(self._create_particle())
    
    def _create_particle(self):
        """Create a single particle for background mist"""
        return {
            'x': random.uniform(0, self.SCREEN_WIDTH),
            'y': random.uniform(0, self.SCREEN_HEIGHT),
            'size': random.uniform(1, 3),
            'speed': random.uniform(0.2, 1),
            'alpha': random.randint(20, 100)
        }
    
    def start_standby_visual(self):
        """Start the mystical floating face standby visual"""
        if self._animation_thread and self._animation_thread.is_alive():
            self.stop_standby_visual()
        
        self._stop_animation = False
        self._animation_thread = threading.Thread(target=self._standby_loop)
        self._animation_thread.daemon = True
        self._animation_thread.start()
        logger.info("Started Pi screen standby visual")
    
    def _standby_loop(self):
        """Main standby animation loop"""
        while not self._stop_animation:
            try:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._stop_animation = True
                        break
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        # Handle touch events
                        self._handle_touch(event.pos)
                
                if self._stop_animation:
                    break
                
                # Update animation state
                self._update_animation()
                
                # Draw everything
                self._draw_frame()
                
                # Cap the frame rate
                self.clock.tick(self.FPS)
                
            except Exception as e:
                logger.error(f"Standby visual error: {e}")
                break
        
        # Clean up when animation ends
        if self.is_initialized and self.screen:
            pygame.quit()
    
    def _update_animation(self):
        """Update animation state"""
        # Update particles
        for particle in self.particles:
            particle['y'] += particle['speed']
            if particle['y'] > self.SCREEN_HEIGHT:
                particle['y'] = 0
                particle['x'] = random.uniform(0, self.SCREEN_WIDTH)
        
        # Update face angle (gentle floating motion)
        self.face_angle += 0.02
        
        # Update eye blink
        self.eye_blink_timer += 1
        if self.eye_blink_timer > 180:  # Blink every ~3 seconds at 60 FPS
            self.eye_blink_duration = 10  # Blink for ~0.17 seconds
            self.eye_blink_timer = 0
        elif self.eye_blink_duration > 0:
            self.eye_blink_duration -= 1
    
    def _draw_frame(self):
        """Draw the complete frame"""
        if not self.is_initialized or not self.screen:
            return
        
        # Clear screen
        self.screen.fill(self.BLACK)
        
        # Draw particles (background mist)
        self._draw_particles()
        
        # Draw mystical floating face
        self._draw_floating_face()
        
        # Update display
        pygame.display.flip()
    
    def _draw_particles(self):
        """Draw background mist particles"""
        for particle in self.particles:
            # Create a surface for the particle with alpha
            particle_surface = pygame.Surface((particle['size'] * 2, particle['size'] * 2), pygame.SRCALPHA)
            color = (100, 150, 200, particle['alpha'])  # Blue-ish mist
            pygame.draw.circle(particle_surface, color, 
                             (particle['size'], particle['size']), particle['size'])
            self.screen.blit(particle_surface, 
                           (particle['x'] - particle['size'], particle['y'] - particle['size']))
    
    def _draw_floating_face(self):
        """Draw the mystical floating face"""
        center_x = self.SCREEN_WIDTH // 2
        center_y = self.SCREEN_HEIGHT // 2
        
        # Add gentle floating motion
        float_offset = math.sin(self.face_angle) * 5
        
        # Draw face outline (ethereal glow effect)
        for i in range(3):
            glow_radius = 80 + i * 5
            glow_alpha = 50 - i * 15
            glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            glow_color = (*self.FACE_COLOR, glow_alpha)
            pygame.draw.circle(glow_surface, glow_color, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surface, 
                           (center_x - glow_radius, center_y - glow_radius + float_offset))
        
        # Draw main face
        face_radius = 60
        pygame.draw.circle(self.screen, self.FACE_COLOR, 
                         (center_x, center_y + int(float_offset)), face_radius)
        
        # Draw eyes
        eye_y = center_y + int(float_offset) - 10
        eye_blink = self.eye_blink_duration > 0
        
        if not eye_blink:
            # Open eyes
            left_eye_x = center_x - 20
            right_eye_x = center_x + 20
            
            # Eye glow effect
            for i in range(2):
                glow_radius = 8 + i * 2
                glow_alpha = 100 - i * 30
                glow_color = (*self.EYE_COLOR_BASE, glow_alpha)
                
                glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, glow_color, (glow_radius, glow_radius), glow_radius)
                
                self.screen.blit(glow_surface, (left_eye_x - glow_radius, eye_y - glow_radius))
                self.screen.blit(glow_surface, (right_eye_x - glow_radius, eye_y - glow_radius))
            
            # Main eye circles
            pygame.draw.circle(self.screen, self.EYE_COLOR_BASE, (left_eye_x, eye_y), 6)
            pygame.draw.circle(self.screen, self.EYE_COLOR_BASE, (right_eye_x, eye_y), 6)
            
            # Eye pupils
            pygame.draw.circle(self.screen, self.BLACK, (left_eye_x, eye_y), 3)
            pygame.draw.circle(self.screen, self.BLACK, (right_eye_x, eye_y), 3)
        else:
            # Closed eyes (blinking)
            pygame.draw.line(self.screen, self.EYE_COLOR_BASE, 
                           (center_x - 25, eye_y), (center_x - 15, eye_y), 3)
            pygame.draw.line(self.screen, self.EYE_COLOR_BASE, 
                           (center_x + 15, eye_y), (center_x + 25, eye_y), 3)
        
        # Draw mystical aura lines
        self._draw_aura_lines(center_x, center_y + int(float_offset))
    
    def _draw_aura_lines(self, center_x: int, center_y: int):
        """Draw mystical aura lines around the face"""
        aura_radius = 100
        num_lines = 8
        
        for i in range(num_lines):
            angle = (self.face_angle * 0.5 + i * (2 * math.pi / num_lines)) % (2 * math.pi)
            start_x = center_x + math.cos(angle) * (aura_radius - 20)
            start_y = center_y + math.sin(angle) * (aura_radius - 20)
            end_x = center_x + math.cos(angle) * aura_radius
            end_y = center_y + math.sin(angle) * aura_radius
            
            # Vary line intensity based on angle
            intensity = abs(math.sin(angle + self.face_angle))
            color = (int(100 + intensity * 100), int(150 + intensity * 100), 255)
            
            pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), 2)
    
    def _handle_touch(self, pos: Tuple[int, int]):
        """Handle touch events on the screen"""
        logger.info(f"Touch detected at position: {pos}")
        # Add any touch interaction logic here
        # For example, could trigger voice activation or show status
    
    def stop_standby_visual(self):
        """Stop the standby visual"""
        self._stop_animation = True
        if self._animation_thread and self._animation_thread.is_alive():
            self._animation_thread.join(timeout=2.0)
        logger.info("Stopped Pi screen standby visual")
    
    def cleanup(self):
        """Clean up screen resources"""
        self.stop_standby_visual()
        if self.is_initialized:
            pygame.quit()
        logger.info("Pi screen cleanup completed")

class HardwareController:
    """Main hardware controller for Raspberry Pi 4 AI Pillar"""
    
    def __init__(self, config: HardwareConfig = None):
        self.config = config or HardwareConfig()
        self.oled = OLEDController(self.config)
        self.rgb_ring = RGBRingController(self.config)
        self.led_strip = LEDStripController(self.config)
        self.current_state = AIState.STARTUP
        self.pi_screen = PiScreenController(self.config)
        
    def initialize(self) -> bool:
        """Initialize all hardware components"""
        logger.info("Initializing Raspberry Pi 4 hardware...")
        
        success = True
        
        # Initialize OLED display
        if not self.oled.initialize():
            logger.warning("OLED initialization failed")
            success = False
        
        # Initialize RGB ring
        if not self.rgb_ring.initialize():
            logger.warning("RGB ring initialization failed")
            success = False
        
        # Initialize LED strip
        if not self.led_strip.initialize():
            logger.warning("LED strip initialization failed")
            success = False
        
        # Initialize Pi screen
        if not self.pi_screen.initialize():
            logger.warning("Pi screen initialization failed")
            success = False
        
        if success:
            logger.info("All hardware components initialized successfully")
        else:
            logger.warning("Some hardware components failed to initialize - running in mixed mode")
        
        return success
    
    def set_ai_state(self, state: AIState):
        """Set AI state and update visual feedback"""
        self.current_state = state
        logger.info(f"Setting AI state: {state.value}")
        
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
    
    def _set_idle_state(self):
        """Set idle state - dim blue RGB ring, LED strip off"""
        self.rgb_ring.stop_animation()
        self.led_strip.stop_animation()
        self.rgb_ring.set_color((0, 0, 50))  # Dim blue
        self.oled.show_text("AI Ready")
    
    def _set_thinking_state(self):
        """Set thinking state - LED strip wave animation"""
        self.rgb_ring.stop_animation()
        self.led_strip.thinking_animation((0, 0, 255))  # Blue wave
        self.oled.show_text("Thinking...")
    
    def _set_speaking_state(self):
        """Set speaking state - RGB ring pulse + OLED wave"""
        self.led_strip.stop_animation()
        self.rgb_ring.pulse_animation((0, 255, 0))  # Green pulse
        self.oled.show_synth_wave()
    
    def _set_listening_state(self):
        """Set listening state - RGB ring pulse"""
        self.led_strip.stop_animation()
        self.rgb_ring.pulse_animation((255, 255, 0))  # Yellow pulse
        self.oled.show_text("Listening...")
    
    def _set_error_state(self):
        """Set error state - red RGB ring"""
        self.rgb_ring.stop_animation()
        self.led_strip.stop_animation()
        self.rgb_ring.set_color((255, 0, 0))  # Red
        self.oled.show_text("Error")
    
    def _set_startup_state(self):
        """Set startup state - rainbow animation"""
        self.rgb_ring.stop_animation()
        self.led_strip.stop_animation()
        self.rgb_ring.set_color((0, 255, 255))  # Cyan
        self.oled.show_text("Starting...")
    
    def start_standby_visual(self):
        """Start the mystical floating face standby visual on Pi screen"""
        if self.pi_screen.is_initialized:
            self.pi_screen.start_standby_visual()
        else:
            logger.warning("Pi screen not initialized - cannot start standby visual")
    
    def stop_standby_visual(self):
        """Stop the standby visual on Pi screen"""
        if self.pi_screen.is_initialized:
            self.pi_screen.stop_standby_visual()
        else:
            logger.warning("Pi screen not initialized - cannot stop standby visual")
    
    def cleanup(self):
        """Clean up hardware resources"""
        logger.info("Cleaning up hardware resources...")
        
        # Stop all animations
        self.rgb_ring.stop_animation()
        self.led_strip.stop_animation()
        
        # Turn off all LEDs
        self.rgb_ring.set_color((0, 0, 0))
        if self.led_strip.is_initialized and self.led_strip.pixels:
            self.led_strip.pixels.fill((0, 0, 0))
            self.led_strip.pixels.show()
        
        # Clear OLED
        self.oled.clear()
        
        # Clean up GPIO
        if GPIO_AVAILABLE:
            try:
                GPIO.cleanup()
            except Exception as e:
                logger.warning(f"GPIO cleanup warning: {e}")
        
        # Clean up Pi screen
        self.pi_screen.cleanup()
        
        logger.info("Hardware cleanup completed")

def get_hardware_controller(config: HardwareConfig = None) -> HardwareController:
    """Get hardware controller instance"""
    return HardwareController(config)

def initialize_hardware(config: HardwareConfig = None) -> bool:
    """Initialize hardware components"""
    controller = get_hardware_controller(config)
    return controller.initialize()

def set_ai_state(state: AIState):
    """Set AI state globally"""
    controller = get_hardware_controller()
    controller.set_ai_state(state)

def start_standby_visual():
    """Start the standby visual globally"""
    controller = get_hardware_controller()
    controller.start_standby_visual()

def stop_standby_visual():
    """Stop the standby visual globally"""
    controller = get_hardware_controller()
    controller.stop_standby_visual()

def cleanup_hardware():
    """Clean up hardware resources"""
    controller = get_hardware_controller()
    controller.cleanup() 