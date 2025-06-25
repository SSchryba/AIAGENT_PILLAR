#!/usr/bin/env python3
"""
Export AI Agent Project to E: Drive for Raspberry Pi 4 Deployment
This script creates a complete, ready-to-deploy package for Raspberry Pi 4.
"""

import os
import shutil
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

class PiExporter:
    def __init__(self, source_dir=".", target_drive="E:"):
        self.source_dir = Path(source_dir).resolve()
        self.target_drive = Path(target_drive)
        self.export_dir = self.target_drive / "AI-Agent-Pi4"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Files to include in the export
        self.include_files = [
            # Core application files
            "agent.py",
            "llm_manager.py", 
            "voice_system.py",
            "memory.py",
            "monitoring.py",
            "agent_config.py",
            "web_interface.py",
            "tools.py",
            "hardware_controller.py",
            "ai_pillar_integration.py",
            
            # Startup scripts
            "start_agent.py",
            "start_ai_pillar.py",
            
            # Configuration files
            "requirements.txt",
            "env_example.txt",
            "Dockerfile",
            "docker-compose.yml",
            
            # Documentation
            "README.md",
            "HARDWARE_INTEGRATION_README.md",
            "VOICE_LLM_SUMMARY.md",
            "OPTIMIZATION_SUMMARY.md",
            "OPTIMIZATION_REPORT.md",
            
            # Test files
            "test_agent.py",
            "test_voice_llm.py", 
            "test_hardware.py",
            "test_pi_screen.py",
            
            # Pi-specific files
            "pi_screen_example.py",
            "pi_screen_standby.py"
        ]
        
        # Directories to create
        self.directories = [
            "logs",
            "db", 
            "models",
            "audio_cache",
            "voice_samples",
            "config",
            "scripts",
            "docs"
        ]
        
        # Files to exclude
        self.exclude_patterns = [
            "__pycache__",
            ".git",
            ".env",
            "*.pyc",
            "*.pyo",
            ".DS_Store",
            "Thumbs.db"
        ]

    def check_target_drive(self):
        """Check if target drive exists and is accessible."""
        if not self.target_drive.exists():
            print(f"❌ Target drive {self.target_drive} does not exist!")
            return False
        
        if not os.access(self.target_drive, os.W_OK):
            print(f"❌ No write access to {self.target_drive}")
            return False
            
        print(f"✅ Target drive {self.target_drive} is accessible")
        return True

    def create_export_structure(self):
        """Create the export directory structure."""
        print(f"\n📁 Creating export structure in {self.export_dir}")
        
        # Create main export directory
        self.export_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        for dir_name in self.directories:
            dir_path = self.export_dir / dir_name
            dir_path.mkdir(exist_ok=True)
            print(f"  📂 Created {dir_path}")

    def copy_files(self):
        """Copy all necessary files to the export directory."""
        print(f"\n📋 Copying files to {self.export_dir}")
        
        copied_count = 0
        missing_count = 0
        
        for file_name in self.include_files:
            source_file = self.source_dir / file_name
            target_file = self.export_dir / file_name
            
            if source_file.exists():
                shutil.copy2(source_file, target_file)
                print(f"  ✅ Copied {file_name}")
                copied_count += 1
            else:
                print(f"  ❌ Missing {file_name}")
                missing_count += 1
        
        print(f"\n📊 File copy summary:")
        print(f"  ✅ Successfully copied: {copied_count} files")
        print(f"  ❌ Missing files: {missing_count} files")

    def create_pi_setup_script(self):
        """Create a Raspberry Pi setup script."""
        setup_script = self.export_dir / "scripts" / "pi_setup.sh"
        
        script_content = f"""#!/bin/bash
# Raspberry Pi 4 Setup Script for AI Agent
# Generated on {self.timestamp}

set -e

echo "🤖 Setting up AI Agent on Raspberry Pi 4..."

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \\
    python3-pip \\
    python3-venv \\
    python3-gpiozero \\
    libopenblas-dev \\
    liblapack-dev \\
    libatlas-base-dev \\
    gfortran \\
    git \\
    curl \\
    wget \\
    build-essential \\
    cmake \\
    pkg-config \\
    libjpeg-dev \\
    libpng-dev \\
    libtiff-dev \\
    libavcodec-dev \\
    libavformat-dev \\
    libswscale-dev \\
    libv4l-dev \\
    libxvidcore-dev \\
    libx264-dev \\
    libgtk-3-dev \\
    libcanberra-gtk3-module \\
    libcanberra-gtk-module \\
    libatlas-base-dev \\
    libhdf5-dev \\
    libhdf5-serial-dev \\
    libhdf5-103 \\
    libqtgui4 \\
    libqtwebkit4 \\
    libqt4-test \\
    python3-pyqt5 \\
    libjasper-dev \\
    libqtcore4 \\
    libqt4-test \\
    libgstreamer1.0-0 \\
    libgstreamer-plugins-base1.0-0 \\
    libgtk-3-0 \\
    libavcodec-dev \\
    libavformat-dev \\
    libswscale-dev \\
    libv4l-dev \\
    libxvidcore-dev \\
    libx264-dev \\
    libjpeg-dev \\
    libpng-dev \\
    libtiff-dev \\
    libatlas-base-dev \\
    gfortran \\
    libhdf5-dev \\
    libhdf5-serial-dev \\
    libhdf5-103 \\
    libqtgui4 \\
    libqtwebkit4 \\
    libqt4-test \\
    python3-pyqt5 \\
    libjasper-dev \\
    libqtcore4 \\
    libqt4-test \\
    libgstreamer1.0-0 \\
    libgstreamer-plugins-base1.0-0 \\
    libgtk-3-0

# Enable hardware interfaces
echo "🔌 Enabling hardware interfaces..."
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_spi 0
sudo raspi-config nonint do_serial 0

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating application directories..."
mkdir -p logs db models audio_cache voice_samples config

# Set up environment
echo "⚙️ Setting up environment..."
if [ ! -f .env ]; then
    cp env_example.txt .env
    echo "📝 Created .env file from template"
fi

# Set permissions
echo "🔐 Setting permissions..."
chmod +x scripts/*.sh
chmod +x start_agent.py
chmod +x start_ai_pillar.py

# Test hardware
echo "🧪 Testing hardware integration..."
python test_hardware.py

echo "✅ Setup complete!"
echo ""
echo "🚀 To start the AI Agent:"
echo "   source venv/bin/activate"
echo "   python web_interface.py"
echo ""
echo "🌐 Access the web interface at: http://localhost:8000"
"""
        
        with open(setup_script, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(setup_script, 0o755)
        print(f"  ✅ Created Pi setup script: {setup_script}")

    def create_startup_scripts(self):
        """Create startup scripts for different scenarios."""
        scripts_dir = self.export_dir / "scripts"
        
        # Auto-start script
        autostart_script = scripts_dir / "autostart.sh"
        autostart_content = f"""#!/bin/bash
# Auto-start script for AI Agent
# Add to /etc/rc.local or systemd service

cd "$(dirname "$0")/.."
source venv/bin/activate
python web_interface.py --host 0.0.0.0 --port 8000
"""
        
        with open(autostart_script, 'w') as f:
            f.write(autostart_content)
        os.chmod(autostart_script, 0o755)
        
        # Development start script
        dev_start_script = scripts_dir / "start_dev.sh"
        dev_start_content = f"""#!/bin/bash
# Development startup script

cd "$(dirname "$0")/.."
source venv/bin/activate
export DEBUG=True
export LOG_LEVEL=DEBUG
python web_interface.py --host 0.0.0.0 --port 8000 --reload
"""
        
        with open(dev_start_script, 'w') as f:
            f.write(dev_start_content)
        os.chmod(dev_start_script, 0o755)
        
        # Hardware test script
        hw_test_script = scripts_dir / "test_hardware.sh"
        hw_test_content = f"""#!/bin/bash
# Hardware test script

cd "$(dirname "$0")/.."
source venv/bin/activate
python test_hardware.py
"""
        
        with open(hw_test_script, 'w') as f:
            f.write(hw_test_content)
        os.chmod(hw_test_script, 0o755)
        
        print(f"  ✅ Created startup scripts in {scripts_dir}")

    def create_systemd_service(self):
        """Create systemd service file for auto-start."""
        service_file = self.export_dir / "scripts" / "ai-agent.service"
        
        service_content = f"""[Unit]
Description=AI Agent Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory={self.export_dir}
Environment=PATH={self.export_dir}/venv/bin
ExecStart={self.export_dir}/venv/bin/python web_interface.py --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        print(f"  ✅ Created systemd service file: {service_file}")

    def create_deployment_guide(self):
        """Create a comprehensive deployment guide."""
        guide_file = self.export_dir / "docs" / "PI_DEPLOYMENT_GUIDE.md"
        
        guide_content = f"""# Raspberry Pi 4 Deployment Guide

## Quick Start

1. **Copy files to Pi**: Transfer all files to your Raspberry Pi 4
2. **Run setup**: `bash scripts/pi_setup.sh`
3. **Start agent**: `python web_interface.py`
4. **Access interface**: http://localhost:8000

## Hardware Requirements

- Raspberry Pi 4 (4GB+ RAM recommended)
- MicroSD card (32GB+ recommended)
- Power supply (5V/3A recommended)
- Touchscreen (optional)
- OLED Display (0.96" I2C)
- RGB LED Ring (16-LED WS2812B)
- LED Strip (60-LED WS2812B)
- Audio system

## Hardware Connections

| Pi Pin | Component | Connection |
|--------|-----------|------------|
| GPIO 18 | RGB Ring Data | WS2812B DIN |
| GPIO 21 | LED Strip Data | WS2812B DIN |
| GPIO 2 (SDA) | OLED SDA | I2C Data |
| GPIO 3 (SCL) | OLED SCL | I2C Clock |
| 3.3V | Power | VCC for all components |
| GND | Ground | GND for all components |

## Configuration

1. Edit `.env` file with your preferences
2. Test hardware: `bash scripts/test_hardware.sh`
3. Configure auto-start (optional): `sudo cp scripts/ai-agent.service /etc/systemd/system/`

## Troubleshooting

- **Hardware not detected**: Check GPIO connections and enable I2C/SPI
- **OLED issues**: Run `i2cdetect -y 1` to scan I2C devices
- **LED issues**: Check power supply and data connections
- **Voice issues**: Check audio system and microphone permissions

## Performance Tips

- Use SSD or fast microSD card
- Enable GPU memory split (128MB+)
- Overclock Pi 4 (optional, at your own risk)
- Use wired network connection

Generated on: {self.timestamp}
"""
        
        with open(guide_file, 'w') as f:
            f.write(guide_content)
        
        print(f"  ✅ Created deployment guide: {guide_file}")

    def create_requirements_pi(self):
        """Create a Pi-optimized requirements file."""
        pi_requirements = self.export_dir / "requirements_pi.txt"
        
        pi_req_content = """# Raspberry Pi 4 Optimized Requirements
# Core dependencies with Pi-specific versions

# Core LangChain dependencies
langchain==0.1.0
langchain-community==0.0.10
langchain-openai==0.0.5

# Vector database and embeddings
chromadb==0.4.22
sentence-transformers==2.2.2

# Web interface
fastapi==0.104.1
uvicorn==0.24.0
jinja2==3.1.2

# Environment and configuration
python-dotenv==1.0.0

# Additional utilities
requests==2.31.0
pydantic==2.5.0
typing-extensions==4.8.0
psutil==5.9.6

# Optional: For enhanced capabilities
beautifulsoup4==4.12.2
python-multipart==0.0.6

# Performance monitoring and optimization
prometheus-client==0.19.0
dataclasses-json==0.6.3

# Testing dependencies
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-mock==3.12.0

# Security and validation
pydantic[email]==2.5.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4

# Enhanced logging and monitoring
structlog==23.2.0
colorama==0.4.6

# Development tools
black==23.11.0
flake8==6.1.0
mypy==1.7.1

# Hugging Face and Multiple LLM Support (Pi-optimized)
transformers==4.35.2
torch==2.1.1+cpu  # CPU-only version for Pi
torchaudio==2.1.1+cpu
accelerate==0.24.1
bitsandbytes==0.41.3
peft==0.7.1
trl==0.7.4
datasets==2.14.6
tokenizers==0.15.0

# Voice Synthesis and Audio Processing (Pi-optimized)
TTS==0.22.0
pyttsx3==2.90
soundfile==0.12.1
librosa==0.10.1
numpy==1.24.3
scipy==1.11.4
webrtcvad==2.0.10
pyaudio==0.2.11
wave==0.0.2

# Audio Streaming and WebRTC
aiortc==1.5.0
websockets==12.0
aiofiles==23.2.1

# Additional AI/ML Libraries (Pi-optimized)
scikit-learn==1.3.2
pandas==2.1.4
matplotlib==3.8.2
seaborn==0.13.0

# Raspberry Pi 4 Hardware Integration Dependencies
# GPIO Control
RPi.GPIO==0.7.1

# OLED Display (I2C)
Pillow==10.1.0  # For OLED display image processing
adafruit-circuitpython-ssd1306==3.3.8  # For OLED display control
adafruit-circuitpython-busdevice==5.2.6  # I2C bus support

# RGB LED Control (WS2812B/NeoPixel)
adafruit-circuitpython-neopixel==6.4.1  # For WS2812B LED control

# Touchscreen Support
pygame==2.5.2  # For touchscreen interface

# WebSocket for real-time communication
websockets==12.0

# Pi-specific optimizations
# Use these for better performance on Pi 4
--extra-index-url https://download.pytorch.org/whl/cpu
"""
        
        with open(pi_requirements, 'w') as f:
            f.write(pi_req_content)
        
        print(f"  ✅ Created Pi-optimized requirements: {pi_requirements}")

    def create_export_summary(self):
        """Create an export summary file."""
        summary_file = self.export_dir / "EXPORT_SUMMARY.md"
        
        summary_content = f"""# AI Agent Export Summary

**Export Date**: {self.timestamp}
**Source Directory**: {self.source_dir}
**Target Directory**: {self.export_dir}

## Included Files

### Core Application
- agent.py - Main agent with multi-LLM and voice support
- llm_manager.py - Multi-LLM manager with Hugging Face models
- voice_system.py - Voice synthesis and streaming system
- memory.py - Enhanced memory with compression
- monitoring.py - Performance monitoring system
- agent_config.py - Comprehensive configuration management
- web_interface.py - FastAPI interface with voice endpoints
- tools.py - Secure tool system
- hardware_controller.py - Raspberry Pi 4 hardware control
- ai_pillar_integration.py - AI Pillar integration module

### Startup Scripts
- start_agent.py - Main agent startup
- start_ai_pillar.py - AI Pillar startup

### Configuration
- requirements.txt - Python dependencies
- requirements_pi.txt - Pi-optimized dependencies
- env_example.txt - Environment template
- Dockerfile - Container configuration
- docker-compose.yml - Docker deployment

### Documentation
- README.md - Main documentation
- HARDWARE_INTEGRATION_README.md - Hardware setup guide
- VOICE_LLM_SUMMARY.md - Voice and LLM documentation
- OPTIMIZATION_SUMMARY.md - Performance optimization guide
- OPTIMIZATION_REPORT.md - Detailed optimization report

### Testing
- test_agent.py - Core functionality tests
- test_voice_llm.py - Voice and LLM specific tests
- test_hardware.py - Hardware integration tests
- test_pi_screen.py - Pi screen tests

### Pi-Specific
- pi_screen_example.py - Pi screen examples
- pi_screen_standby.py - Pi screen standby mode

### Scripts
- scripts/pi_setup.sh - Complete Pi setup script
- scripts/autostart.sh - Auto-start script
- scripts/start_dev.sh - Development startup
- scripts/test_hardware.sh - Hardware testing
- scripts/ai-agent.service - Systemd service file

### Documentation
- docs/PI_DEPLOYMENT_GUIDE.md - Comprehensive deployment guide

## Deployment Instructions

1. **Transfer to Pi**: Copy all files to Raspberry Pi 4
2. **Run Setup**: `bash scripts/pi_setup.sh`
3. **Test Hardware**: `bash scripts/test_hardware.sh`
4. **Start Agent**: `python web_interface.py`
5. **Access Interface**: http://localhost:8000

## Hardware Requirements

- Raspberry Pi 4 (4GB+ RAM recommended)
- Touchscreen (optional)
- OLED Display (0.96" I2C)
- RGB LED Ring (16-LED WS2812B)
- LED Strip (60-LED WS2812B)
- Audio system

## Performance Notes

- Use Pi-optimized requirements: `pip install -r requirements_pi.txt`
- Enable GPU memory split (128MB+)
- Use wired network connection
- Consider SSD for better performance

## Support

For issues and questions, refer to:
- README.md - Main documentation
- docs/PI_DEPLOYMENT_GUIDE.md - Deployment guide
- HARDWARE_INTEGRATION_README.md - Hardware setup
"""
        
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        
        print(f"  ✅ Created export summary: {summary_file}")

    def export(self):
        """Perform the complete export process."""
        print("🚀 Starting AI Agent export to Raspberry Pi 4...")
        print(f"📂 Source: {self.source_dir}")
        print(f"🎯 Target: {self.export_dir}")
        
        # Check target drive
        if not self.check_target_drive():
            return False
        
        try:
            # Create export structure
            self.create_export_structure()
            
            # Copy files
            self.copy_files()
            
            # Create Pi-specific files
            print(f"\n🔧 Creating Pi-specific files...")
            self.create_pi_setup_script()
            self.create_startup_scripts()
            self.create_systemd_service()
            self.create_deployment_guide()
            self.create_requirements_pi()
            self.create_export_summary()
            
            # Calculate export size
            total_size = sum(f.stat().st_size for f in self.export_dir.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            
            print(f"\n✅ Export completed successfully!")
            print(f"📊 Export size: {size_mb:.1f} MB")
            print(f"📁 Export location: {self.export_dir}")
            print(f"\n🚀 Next steps:")
            print(f"   1. Transfer files to Raspberry Pi 4")
            print(f"   2. Run: bash scripts/pi_setup.sh")
            print(f"   3. Test hardware: bash scripts/test_hardware.sh")
            print(f"   4. Start agent: python web_interface.py")
            print(f"   5. Access: http://localhost:8000")
            
            return True
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return False

def main():
    """Main function to run the export."""
    exporter = PiExporter()
    success = exporter.export()
    
    if success:
        print(f"\n🎉 Export to {exporter.target_drive} completed successfully!")
        sys.exit(0)
    else:
        print(f"\n💥 Export failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 