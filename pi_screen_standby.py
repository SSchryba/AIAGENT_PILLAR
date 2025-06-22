#!/usr/bin/env python3
"""
Pi Screen Standby Visual
Creates a mystical floating face standby visual for Raspberry Pi 4 touchscreen.
"""

import time
import random
import math
import logging
import threading
from typing import Tuple, Optional

# Pygame imports
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logging.warning("Pygame not available - Pi screen functionality disabled")

logger = logging.getLogger(__name__)

class PiScreenStandby:
    """Standalone Pi screen standby visual controller"""
    
    def __init__(self, screen_width: int = 800, screen_height: int = 480, fps: int = 60):
        self.SCREEN_WIDTH = screen_width
        self.SCREEN_HEIGHT = screen_height
        self.FPS = fps
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.FACE_COLOR = (120, 180, 255)
        self.EYE_COLOR_BASE = (0, 255, 255)
        
        # Pygame objects
        self.screen = None
        self.clock = None
        self.is_initialized = False
        
        # Animation state
        self.particles = []
        self.face_angle = 0
        self.eye_blink_timer = 0
        self.eye_blink_duration = 0
        
        # Threading
        self._animation_thread = None
        self._stop_animation = False
    
    def initialize(self) -> bool:
        """Initialize the Pi screen standby visual"""
        if not PYGAME_AVAILABLE:
            logger.error("Pygame not available - cannot initialize Pi screen")
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
            logger.info("Pi screen standby visual initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Pi screen standby visual: {e}")
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
    
    def start(self):
        """Start the standby visual"""
        if not self.is_initialized:
            logger.error("Pi screen not initialized - cannot start standby visual")
            return False
        
        if self._animation_thread and self._animation_thread.is_alive():
            self.stop()
        
        self._stop_animation = False
        self._animation_thread = threading.Thread(target=self._animation_loop)
        self._animation_thread.daemon = True
        self._animation_thread.start()
        
        logger.info("Started Pi screen standby visual")
        return True
    
    def _animation_loop(self):
        """Main animation loop"""
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
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self._stop_animation = True
                            break
                
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
    
    def stop(self):
        """Stop the standby visual"""
        self._stop_animation = True
        if self._animation_thread and self._animation_thread.is_alive():
            self._animation_thread.join(timeout=2.0)
        logger.info("Stopped Pi screen standby visual")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop()
        if self.is_initialized:
            pygame.quit()
        logger.info("Pi screen standby visual cleanup completed")

def create_standby_visual(screen_width: int = 800, screen_height: int = 480, fps: int = 60) -> PiScreenStandby:
    """Create and return a Pi screen standby visual instance"""
    return PiScreenStandby(screen_width, screen_height, fps)

def run_standby_visual(screen_width: int = 800, screen_height: int = 480, fps: int = 60):
    """Run the standby visual (blocking function)"""
    standby = create_standby_visual(screen_width, screen_height, fps)
    
    if not standby.initialize():
        logger.error("Failed to initialize standby visual")
        return False
    
    try:
        standby.start()
        # The animation will run until stopped or interrupted
        while standby._animation_thread and standby._animation_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Standby visual interrupted by user")
    finally:
        standby.cleanup()
    
    return True

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the standby visual
    print("Starting Pi screen standby visual...")
    print("Press Ctrl+C to stop or ESC key to exit")
    
    success = run_standby_visual()
    
    if success:
        print("Standby visual completed successfully")
    else:
        print("Standby visual failed to start") 