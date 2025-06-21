#!/usr/bin/env python3
"""
Enhanced Web Interface for AI Agent
Supports multiple LLM models, voice synthesis, and real-time features.
"""

import os
import logging
import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import time

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
import uvicorn
from jinja2 import Environment, FileSystemLoader
import psutil

from agent import get_agent
from memory import get_memory
from monitoring import get_monitoring_system
from llm_manager import get_llm_manager
from voice_system import get_voice_system, VoiceConfig, VoiceEngine, get_voice_stream
import agent_config as config

# Configure logging
logging.basicConfig(level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')))
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Agent API",
    description="A sophisticated AI agent with multiple LLM support and voice synthesis",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.SECURITY_SETTINGS["allowed_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        client_requests = self.requests[client_ip]
        
        # Remove old requests outside the window
        client_requests[:] = [req_time for req_time in client_requests 
                            if now - req_time < self.window_seconds]
        
        # Check if under limit
        if len(client_requests) < self.max_requests:
            client_requests.append(now)
            return True
        return False

rate_limiter = RateLimiter(
    max_requests=config.WEB_SETTINGS["rate_limit"],
    window_seconds=60
)

# Request models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    generate_voice: bool = Field(default=True)
    voice_name: Optional[str] = Field(default=None)
    emotion: Optional[str] = Field(default=None)
    model_name: Optional[str] = Field(default=None)

class VoiceRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice_name: Optional[str] = Field(default=None)
    emotion: Optional[str] = Field(default=None)

class ModelSwitchRequest(BaseModel):
    model_name: str = Field(..., min_length=1)

class VoiceConfigRequest(BaseModel):
    voice_name: str = Field(..., min_length=1)
    speed: Optional[float] = Field(default=1.0, ge=0.1, le=3.0)
    pitch: Optional[float] = Field(default=1.0, ge=0.1, le=3.0)
    volume: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)

# Global instances
agent = None
memory = None
monitoring = None
llm_manager = None
voice_system = None
voice_stream = None

def get_client_ip(request: Request) -> str:
    """Get client IP address."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0]
    return request.client.host

def check_rate_limit(request: Request):
    """Check rate limit for the request."""
    client_ip = get_client_ip(request)
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global agent, memory, monitoring, llm_manager, voice_system, voice_stream
    
    try:
        logger.info("Initializing AI Agent components...")
        
        # Initialize agent
        agent = get_agent(
            model_name=config.MODEL_CONFIG["default_model"],
            enable_voice=config.VOICE_SETTINGS.get("enable_voice", True)
        )
        
        # Initialize other components
        memory = get_memory()
        monitoring = get_monitoring_system()
        llm_manager = get_llm_manager()
        
        # Initialize voice system
        if config.VOICE_SETTINGS.get("enable_voice", True):
            voice_config = VoiceConfig(
                engine=VoiceEngine.TTS,
                voice_name=config.VOICE_SETTINGS["voice_name"],
                quality=config.VOICE_SETTINGS["quality"],
                speed=config.VOICE_SETTINGS["speed"],
                volume=config.VOICE_SETTINGS["volume"],
                language=config.VOICE_SETTINGS["language"]
            )
            voice_system = get_voice_system(voice_config)
            voice_stream = get_voice_stream(voice_system)
            
            # Start voice streaming if enabled
            if config.VOICE_SETTINGS.get("enable_real_time", False):
                await voice_stream.start_streaming(
                    port=config.VOICE_SETTINGS["voice_streaming_port"]
                )
        
        logger.info("AI Agent components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global voice_stream
    if voice_stream:
        await voice_stream.stop_streaming()
    logger.info("AI Agent shutdown complete")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": config.AGENT_VERSION,
            "agent_name": config.AGENT_NAME,
            "components": {
                "agent": agent is not None,
                "memory": memory is not None,
                "monitoring": monitoring is not None,
                "llm_manager": llm_manager is not None,
                "voice_system": voice_system is not None
            }
        }
        
        # Check system resources
        status["system"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
        
        return status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

# Main chat endpoint
@app.post("/api/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Process a chat message with optional voice synthesis."""
    try:
        # Check rate limit
        check_rate_limit(request)
        
        # Switch model if requested
        if request.model_name and request.model_name != agent.get_current_model():
            if not agent.switch_model(request.model_name):
                raise HTTPException(status_code=400, detail="Invalid model name")
        
        # Process the message
        start_time = time.time()
        response_data = agent.chat(
            message=request.message,
            generate_voice=request.generate_voice
        )
        processing_time = time.time() - start_time
        
        # Record metrics
        if monitoring:
            monitoring.record_request(
                endpoint="/api/chat",
                response_time=processing_time,
                success=not response_data.get("error", False)
            )
        
        # Add metadata
        response_data.update({
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "model_used": agent.get_current_model()
        })
        
        return response_data
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        if monitoring:
            monitoring.record_error("/api/chat", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Voice synthesis endpoint
@app.post("/api/voice/synthesize")
async def synthesize_voice(request: VoiceRequest):
    """Synthesize text to speech."""
    try:
        if not voice_system:
            raise HTTPException(status_code=503, detail="Voice system not available")
        
        audio_data = voice_system.synthesize_text(
            text=request.text,
            voice_name=request.voice_name,
            emotion=request.emotion
        )
        
        if not audio_data:
            raise HTTPException(status_code=500, detail="Failed to synthesize voice")
        
        return StreamingResponse(
            iter([audio_data]),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
        
    except Exception as e:
        logger.error(f"Voice synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Voice streaming endpoint
@app.websocket("/ws/voice")
async def voice_websocket(websocket):
    """WebSocket endpoint for real-time voice streaming."""
    try:
        await websocket.accept()
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "synthesize":
                text = message.get("text", "")
                voice_name = message.get("voice_name")
                emotion = message.get("emotion")
                
                if voice_system:
                    audio_data = voice_system.synthesize_text(text, voice_name, emotion)
                    await websocket.send_bytes(audio_data)
                else:
                    await websocket.send_text(json.dumps({"error": "Voice system not available"}))
                    
    except Exception as e:
        logger.error(f"Voice WebSocket error: {e}")

# Model management endpoints
@app.get("/api/models")
async def get_models():
    """Get available LLM models."""
    try:
        if not llm_manager:
            raise HTTPException(status_code=503, detail="LLM manager not available")
        
        models = llm_manager.get_available_models()
        current_model = llm_manager.get_current_model()
        performance = llm_manager.get_all_performance_metrics()
        
        return {
            "available_models": models,
            "current_model": current_model,
            "performance": performance
        }
        
    except Exception as e:
        logger.error(f"Get models error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/switch")
async def switch_model(request: ModelSwitchRequest):
    """Switch to a different LLM model."""
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not available")
        
        success = agent.switch_model(request.model_name)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to switch model")
        
        return {
            "success": True,
            "current_model": agent.get_current_model(),
            "message": f"Switched to model: {request.model_name}"
        }
        
    except Exception as e:
        logger.error(f"Switch model error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Voice management endpoints
@app.get("/api/voice/voices")
async def get_voices():
    """Get available voices."""
    try:
        if not voice_system:
            raise HTTPException(status_code=503, detail="Voice system not available")
        
        voices = voice_system.get_available_voices()
        voice_info = voice_system.get_engine_info()
        
        return {
            "available_voices": voices,
            "voice_info": voice_info
        }
        
    except Exception as e:
        logger.error(f"Get voices error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/set")
async def set_voice(request: VoiceConfigRequest):
    """Set voice configuration."""
    try:
        if not voice_system:
            raise HTTPException(status_code=503, detail="Voice system not available")
        
        success = voice_system.set_voice(request.voice_name)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to set voice")
        
        return {
            "success": True,
            "voice_name": request.voice_name,
            "message": f"Voice set to: {request.voice_name}"
        }
        
    except Exception as e:
        logger.error(f"Set voice error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Memory management endpoints
@app.get("/api/memory/summary")
async def get_memory_summary():
    """Get memory summary."""
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not available")
        
        return agent.get_memory_summary()
        
    except Exception as e:
        logger.error(f"Memory summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/clear")
async def clear_memory():
    """Clear memory."""
    try:
        if not memory:
            raise HTTPException(status_code=503, detail="Memory not available")
        
        memory.clear_memory()
        return {"success": True, "message": "Memory cleared"}
        
    except Exception as e:
        logger.error(f"Clear memory error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/export")
async def export_memory():
    """Export memory to file."""
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not available")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"memory_export_{timestamp}.json"
        
        success = agent.export_conversation(filename)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to export memory")
        
        return {
            "success": True,
            "filename": filename,
            "message": "Memory exported successfully"
        }
        
    except Exception as e:
        logger.error(f"Export memory error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Cache management endpoints
@app.post("/api/cache/clear")
async def clear_cache():
    """Clear response cache."""
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not available")
        
        agent.clear_cache()
        return {"success": True, "message": "Cache cleared"}
        
    except Exception as e:
        logger.error(f"Clear cache error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System information endpoint
@app.get("/api/system-info")
async def get_system_info():
    """Get system information."""
    try:
        info = {
            "agent": {
                "name": config.AGENT_NAME,
                "version": config.AGENT_VERSION,
                "current_model": agent.get_current_model() if agent else None,
                "voice_enabled": voice_system is not None
            },
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "uptime": time.time() - psutil.boot_time()
            },
            "configuration": {
                "web_settings": config.WEB_SETTINGS,
                "voice_settings": config.VOICE_SETTINGS,
                "model_config": config.MODEL_CONFIG
            }
        }
        
        return info
        
    except Exception as e:
        logger.error(f"System info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Monitoring endpoints
@app.get("/api/metrics/summary")
async def get_metrics_summary():
    """Get monitoring metrics summary."""
    try:
        if not monitoring:
            raise HTTPException(status_code=503, detail="Monitoring not available")
        
        return monitoring.get_metrics_summary()
        
    except Exception as e:
        logger.error(f"Metrics summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics/detailed")
async def get_metrics_detailed(hours: int = 1):
    """Get detailed monitoring metrics."""
    try:
        if not monitoring:
            raise HTTPException(status_code=503, detail="Monitoring not available")
        
        return monitoring.get_detailed_metrics(hours)
        
    except Exception as e:
        logger.error(f"Detailed metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/metrics/export")
async def export_metrics():
    """Export monitoring metrics."""
    try:
        if not monitoring:
            raise HTTPException(status_code=503, detail="Monitoring not available")
        
        filename = monitoring.export_metrics()
        return {
            "success": True,
            "filename": filename,
            "message": "Metrics exported successfully"
        }
        
    except Exception as e:
        logger.error(f"Export metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/metrics/reset")
async def reset_metrics():
    """Reset monitoring metrics."""
    try:
        if not monitoring:
            raise HTTPException(status_code=503, detail="Monitoring not available")
        
        monitoring.reset_metrics()
        return {"success": True, "message": "Metrics reset successfully"}
        
    except Exception as e:
        logger.error(f"Reset metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Configuration endpoint
@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    try:
        return config.get_config()
        
    except Exception as e:
        logger.error(f"Get config error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Web interface
@app.get("/", response_class=HTMLResponse)
async def web_interface():
    """Serve the web interface."""
    try:
        # Create a simple HTML interface
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{config.AGENT_NAME} - AI Agent</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .chat-container {{
                    background: white;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .message {{
                    margin: 10px 0;
                    padding: 10px;
                    border-radius: 5px;
                }}
                .user-message {{
                    background-color: #e3f2fd;
                    text-align: right;
                }}
                .agent-message {{
                    background-color: #f3e5f5;
                }}
                .input-container {{
                    display: flex;
                    gap: 10px;
                    margin-top: 20px;
                }}
                #messageInput {{
                    flex: 1;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                button {{
                    padding: 10px 20px;
                    background-color: #2196f3;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                }}
                button:hover {{
                    background-color: #1976d2;
                }}
                .controls {{
                    margin: 20px 0;
                    display: flex;
                    gap: 10px;
                    flex-wrap: wrap;
                }}
                .status {{
                    background-color: #e8f5e8;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <h1>🤖 {config.AGENT_NAME} - AI Agent v{config.AGENT_VERSION}</h1>
            
            <div class="status">
                <strong>Status:</strong> 
                <span id="status">Connecting...</span>
                <br>
                <strong>Current Model:</strong> 
                <span id="currentModel">Loading...</span>
                <br>
                <strong>Voice:</strong> 
                <span id="voiceStatus">Loading...</span>
            </div>
            
            <div class="controls">
                <select id="modelSelect">
                    <option value="">Select Model</option>
                </select>
                <select id="voiceSelect">
                    <option value="">Select Voice</option>
                </select>
                <label>
                    <input type="checkbox" id="voiceEnabled" checked> Enable Voice
                </label>
                <button onclick="clearChat()">Clear Chat</button>
                <button onclick="exportChat()">Export Chat</button>
            </div>
            
            <div class="chat-container">
                <div id="chatMessages"></div>
                <div class="input-container">
                    <input type="text" id="messageInput" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
                    <button onclick="sendMessage()">Send</button>
                    <button onclick="speakMessage()">🎤 Speak</button>
                </div>
            </div>
            
            <script>
                let currentModel = '';
                let currentVoice = '';
                
                // Initialize
                async function initialize() {{
                    await updateStatus();
                    await loadModels();
                    await loadVoices();
                }}
                
                async function updateStatus() {{
                    try {{
                        const response = await fetch('/health');
                        const data = await response.json();
                        document.getElementById('status').textContent = data.status;
                        document.getElementById('currentModel').textContent = data.components.agent ? 'Connected' : 'Disconnected';
                        document.getElementById('voiceStatus').textContent = data.components.voice_system ? 'Available' : 'Not Available';
                    }} catch (error) {{
                        document.getElementById('status').textContent = 'Error';
                    }}
                }}
                
                async function loadModels() {{
                    try {{
                        const response = await fetch('/api/models');
                        const data = await response.json();
                        const select = document.getElementById('modelSelect');
                        select.innerHTML = '<option value="">Select Model</option>';
                        
                        data.available_models.forEach(model => {{
                            const option = document.createElement('option');
                            option.value = model;
                            option.textContent = model;
                            if (model === data.current_model) {{
                                option.selected = true;
                                currentModel = model;
                            }}
                            select.appendChild(option);
                        }});
                    }} catch (error) {{
                        console.error('Failed to load models:', error);
                    }}
                }}
                
                async function loadVoices() {{
                    try {{
                        const response = await fetch('/api/voice/voices');
                        const data = await response.json();
                        const select = document.getElementById('voiceSelect');
                        select.innerHTML = '<option value="">Select Voice</option>';
                        
                        Object.keys(data.available_voices).forEach(voice => {{
                            const option = document.createElement('option');
                            option.value = voice;
                            option.textContent = voice;
                            select.appendChild(option);
                        }});
                    }} catch (error) {{
                        console.error('Failed to load voices:', error);
                    }}
                }}
                
                async function sendMessage() {{
                    const input = document.getElementById('messageInput');
                    const message = input.value.trim();
                    if (!message) return;
                    
                    const voiceEnabled = document.getElementById('voiceEnabled').checked;
                    
                    addMessage('user', message);
                    input.value = '';
                    
                    try {{
                        const response = await fetch('/api/chat', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify({{
                                message: message,
                                generate_voice: voiceEnabled,
                                model_name: currentModel
                            }})
                        }});
                        
                        const data = await response.json();
                        addMessage('agent', data.text);
                        
                        if (data.voice_audio && voiceEnabled) {{
                            playAudio(data.voice_audio);
                        }}
                    }} catch (error) {{
                        addMessage('agent', 'Sorry, I encountered an error. Please try again.');
                        console.error('Chat error:', error);
                    }}
                }}
                
                function addMessage(sender, text) {{
                    const messagesDiv = document.getElementById('chatMessages');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `message ${{sender}}-message`;
                    messageDiv.textContent = text;
                    messagesDiv.appendChild(messageDiv);
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                }}
                
                function playAudio(audioData) {{
                    // Convert base64 audio data to blob and play
                    const audioBlob = new Blob([new Uint8Array(audioData)], {{ type: 'audio/wav' }});
                    const audioUrl = URL.createObjectURL(audioBlob);
                    const audio = new Audio(audioUrl);
                    audio.play();
                }}
                
                function handleKeyPress(event) {{
                    if (event.key === 'Enter') {{
                        sendMessage();
                    }}
                }}
                
                function clearChat() {{
                    document.getElementById('chatMessages').innerHTML = '';
                }}
                
                async function exportChat() {{
                    try {{
                        const response = await fetch('/api/memory/export', {{ method: 'POST' }});
                        const data = await response.json();
                        if (data.success) {{
                            alert('Chat exported successfully!');
                        }}
                    }} catch (error) {{
                        alert('Failed to export chat');
                    }}
                }}
                
                // Model switching
                document.getElementById('modelSelect').addEventListener('change', async function() {{
                    const model = this.value;
                    if (model && model !== currentModel) {{
                        try {{
                            const response = await fetch('/api/models/switch', {{
                                method: 'POST',
                                headers: {{ 'Content-Type': 'application/json' }},
                                body: JSON.stringify({{ model_name: model }})
                            }});
                            const data = await response.json();
                            if (data.success) {{
                                currentModel = model;
                                alert(`Switched to model: ${{model}}`);
                            }}
                        }} catch (error) {{
                            alert('Failed to switch model');
                        }}
                    }}
                }});
                
                // Voice switching
                document.getElementById('voiceSelect').addEventListener('change', async function() {{
                    const voice = this.value;
                    if (voice && voice !== currentVoice) {{
                        try {{
                            const response = await fetch('/api/voice/set', {{
                                method: 'POST',
                                headers: {{ 'Content-Type': 'application/json' }},
                                body: JSON.stringify({{ voice_name: voice }})
                            }});
                            const data = await response.json();
                            if (data.success) {{
                                currentVoice = voice;
                                alert(`Voice set to: ${{voice}}`);
                            }}
                        }} catch (error) {{
                            alert('Failed to set voice');
                        }}
                    }}
                }});
                
                // Initialize on load
                window.onload = initialize;
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"Web interface error: {e}")
        raise HTTPException(status_code=500, detail="Failed to load web interface")

if __name__ == "__main__":
    uvicorn.run(
        "web_interface:app",
        host=config.WEB_SETTINGS["host"],
        port=config.WEB_SETTINGS["port"],
        reload=config.WEB_SETTINGS["debug"]
    ) 