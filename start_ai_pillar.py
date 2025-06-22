#!/usr/bin/env python3
"""
AI Pillar Startup Script for Raspberry Pi 4
Initializes hardware and starts the web interface
"""

import asyncio
import logging
import sys
import signal
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from ai_pillar_integration import PillarConfig, PillarMode, initialize_pillar, shutdown_pillar
from web_interface import app
import uvicorn

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_pillar.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AIPillarService:
    """AI Pillar service for Raspberry Pi 4"""
    
    def __init__(self):
        self.is_running = False
        self.server = None
        
    async def initialize(self):
        """Initialize AI Pillar"""
        try:
            logger.info("Starting AI Pillar on Raspberry Pi 4...")
            
            # Check if running on Raspberry Pi
            if not self._is_raspberry_pi():
                logger.warning("Not running on Raspberry Pi - some features may not work")
            
            # Initialize AI Pillar
            config = PillarConfig(
                mode=PillarMode.STANDALONE,
                enable_voice=True,
                enable_visual_feedback=True,
                enable_touchscreen=True
            )
            
            success = await initialize_pillar(config)
            if not success:
                logger.error("Failed to initialize AI Pillar")
                return False
            
            logger.info("AI Pillar initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Pillar: {e}")
            return False
    
    def _is_raspberry_pi(self):
        """Check if running on Raspberry Pi"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                return 'Raspberry Pi' in f.read()
        except:
            return False
    
    async def start_server(self, host="0.0.0.0", port=8000):
        """Start the web server"""
        try:
            logger.info(f"Starting web server on {host}:{port}")
            
            # Configure uvicorn
            config = uvicorn.Config(
                app=app,
                host=host,
                port=port,
                log_level="info",
                access_log=True,
                workers=1  # Single worker for Raspberry Pi
            )
            
            self.server = uvicorn.Server(config)
            self.is_running = True
            
            # Start server
            await self.server.serve()
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            self.is_running = False
    
    async def shutdown(self):
        """Shutdown AI Pillar"""
        try:
            logger.info("Shutting down AI Pillar...")
            
            self.is_running = False
            
            # Stop server if running
            if self.server:
                self.server.should_exit = True
            
            # Shutdown AI Pillar
            await shutdown_pillar()
            
            logger.info("AI Pillar shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Global service instance
service = AIPillarService()

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    asyncio.create_task(service.shutdown())

async def main():
    """Main startup function"""
    try:
        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Initialize AI Pillar
        if not await service.initialize():
            logger.error("Failed to initialize AI Pillar")
            sys.exit(1)
        
        # Start web server
        await service.start_server()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await service.shutdown()

def run_with_sudo_check():
    """Check if running with sudo and provide guidance"""
    if os.geteuid() == 0:
        logger.warning("Running as root - this is not recommended for security reasons")
        logger.info("Consider running as a regular user and adding to gpio group")
        return True
    else:
        # Check if user is in gpio group
        try:
            import grp
            gpio_group = grp.getgrnam('gpio')
            if os.getlogin() in gpio_group.gr_mem:
                return True
            else:
                logger.warning("User not in gpio group - hardware may not work")
                logger.info("Run: sudo usermod -a -G gpio $USER")
                return False
        except:
            logger.warning("Could not check gpio group membership")
            return True

if __name__ == "__main__":
    # Check permissions
    run_with_sudo_check()
    
    # Run the main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1) 