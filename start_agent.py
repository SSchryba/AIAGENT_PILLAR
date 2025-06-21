#!/usr/bin/env python3
"""
AI Agent Startup Script
Handles environment setup and launches the AI agent with proper configuration.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from dotenv import load_dotenv

def setup_logging():
    """Configure logging for the startup process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'fastapi', 'uvicorn', 'langchain', 'chromadb', 
        'sentence-transformers', 'python-dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def setup_environment():
    """Set up environment variables and configuration."""
    logger = logging.getLogger(__name__)
    
    # Check if .env file exists
    env_file = Path('.env')
    env_example = Path('env_example.txt')
    
    if not env_file.exists():
        if env_example.exists():
            print("📝 Creating .env file from template...")
            with open(env_example, 'r') as f:
                content = f.read()
            with open(env_file, 'w') as f:
                f.write(content)
            print("✅ .env file created. Please edit it with your configuration.")
        else:
            print("⚠️  No .env file found. Creating basic configuration...")
            basic_env = """# AI Agent Configuration
AGENT_NAME=Lila
AGENT_TONE=Curious, witty, emotionally intelligent
MEMORY_PERSIST_DIR=./db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HOST=localhost
PORT=8000
DEBUG=True
LOG_LEVEL=INFO
LOG_FILE=agent.log
"""
            with open(env_file, 'w') as f:
                f.write(basic_env)
    
    # Load environment variables
    load_dotenv()
    
    # Create necessary directories
    directories = ['db', 'logs', 'templates']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Environment setup complete")

def check_openai_key():
    """Check if OpenAI API key is configured."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  OpenAI API key not found in .env file")
        print("   The agent will use fallback mode or may not function properly")
        print("   Add your OpenAI API key to the .env file to enable full functionality")
        return False
    print("✅ OpenAI API key configured")
    return True

def start_agent():
    """Start the AI agent."""
    logger = logging.getLogger(__name__)
    
    try:
        print("\n🚀 Starting AI Agent...")
        print("=" * 50)
        
        # Import and start the web interface
        from web_interface import app
        import uvicorn
        import agent_config as config
        
        host = config.WEB_SETTINGS["host"]
        port = config.WEB_SETTINGS["port"]
        
        print(f"📍 Agent will be available at: http://{host}:{port}")
        print(f"🤖 Agent Name: {config.AGENT_NAME}")
        print(f"🔧 Debug Mode: {config.WEB_SETTINGS['debug']}")
        print("\nPress Ctrl+C to stop the agent")
        print("=" * 50)
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=config.WEB_SETTINGS["debug"]
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Agent stopped by user")
    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        print(f"❌ Error starting agent: {e}")
        sys.exit(1)

def main():
    """Main startup function."""
    print("🤖 AI Agent Startup")
    print("=" * 30)
    
    # Setup logging
    logger = setup_logging()
    
    # Check prerequisites
    check_python_version()
    
    if not check_dependencies():
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Check API key
    check_openai_key()
    
    # Start the agent
    start_agent()

if __name__ == "__main__":
    main() 