import os
import sys
import platform
import psutil
import requests
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class SystemTools:
    """Tools for system information and operations."""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information."""
        try:
            return {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "architecture": platform.architecture()[0],
                "processor": platform.processor(),
                "hostname": platform.node(),
                "python_version": sys.version,
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_usage": psutil.disk_usage('/')._asdict(),
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def get_process_info() -> List[Dict[str, Any]]:
        """Get information about running processes."""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)[:10]
        except Exception as e:
            logger.error(f"Error getting process info: {e}")
            return []
    
    @staticmethod
    def get_network_info() -> Dict[str, Any]:
        """Get network information."""
        try:
            return {
                "interfaces": psutil.net_if_addrs(),
                "connections": len(psutil.net_connections()),
                "io_counters": psutil.net_io_counters()._asdict()
            }
        except Exception as e:
            logger.error(f"Error getting network info: {e}")
            return {"error": str(e)}

class FileTools:
    """Tools for file operations."""
    
    @staticmethod
    def list_directory(path: str = ".") -> Dict[str, Any]:
        """List contents of a directory."""
        try:
            path_obj = Path(path).resolve()
            if not path_obj.exists():
                return {"error": f"Path {path} does not exist"}
            
            contents = {
                "path": str(path_obj),
                "files": [],
                "directories": [],
                "total_size": 0
            }
            
            for item in path_obj.iterdir():
                try:
                    if item.is_file():
                        size = item.stat().st_size
                        contents["files"].append({
                            "name": item.name,
                            "size": size,
                            "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                        })
                        contents["total_size"] += size
                    elif item.is_dir():
                        contents["directories"].append({
                            "name": item.name,
                            "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                        })
                except (PermissionError, OSError):
                    continue
            
            return contents
        except Exception as e:
            logger.error(f"Error listing directory {path}: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def read_file(file_path: str, max_size: int = 1024 * 1024) -> Dict[str, Any]:
        """Read a file's contents."""
        try:
            path_obj = Path(file_path).resolve()
            if not path_obj.exists():
                return {"error": f"File {file_path} does not exist"}
            
            if not path_obj.is_file():
                return {"error": f"{file_path} is not a file"}
            
            size = path_obj.stat().st_size
            if size > max_size:
                return {"error": f"File too large ({size} bytes), max allowed: {max_size}"}
            
            with open(path_obj, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                "path": str(path_obj),
                "size": size,
                "content": content,
                "lines": len(content.splitlines())
            }
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def search_files(directory: str, pattern: str) -> List[str]:
        """Search for files matching a pattern."""
        try:
            path_obj = Path(directory).resolve()
            if not path_obj.exists():
                return []
            
            matches = []
            for item in path_obj.rglob(pattern):
                if item.is_file():
                    matches.append(str(item))
            
            return matches
        except Exception as e:
            logger.error(f"Error searching files: {e}")
            return []

class WebTools:
    """Tools for web operations."""
    
    @staticmethod
    def check_url(url: str) -> Dict[str, Any]:
        """Check if a URL is accessible."""
        try:
            response = requests.get(url, timeout=10)
            return {
                "url": url,
                "status_code": response.status_code,
                "accessible": response.status_code == 200,
                "content_type": response.headers.get('content-type', ''),
                "content_length": len(response.content)
            }
        except requests.exceptions.RequestException as e:
            return {
                "url": url,
                "error": str(e),
                "accessible": False
            }
    
    @staticmethod
    def get_weather(city: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Get weather information for a city."""
        # This is a placeholder - you would need to implement with a real weather API
        return {
            "city": city,
            "message": "Weather API not configured. Please add API key to use this feature."
        }

class CommandTools:
    """Tools for executing system commands."""
    
    @staticmethod
    def execute_command(command: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute a system command safely."""
        try:
            # Basic security check - prevent dangerous commands
            dangerous_commands = ['rm -rf', 'format', 'del', 'shutdown', 'reboot']
            if any(dangerous in command.lower() for dangerous in dangerous_commands):
                return {"error": "Command not allowed for security reasons"}
            
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                "command": command,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {"error": f"Command timed out after {timeout} seconds"}
        except Exception as e:
            logger.error(f"Error executing command {command}: {e}")
            return {"error": str(e)}

class UtilityTools:
    """General utility tools."""
    
    @staticmethod
    def get_current_time() -> str:
        """Get current timestamp."""
        return datetime.now().isoformat()
    
    @staticmethod
    def calculate(expression: str) -> Dict[str, Any]:
        """Safely evaluate a mathematical expression."""
        try:
            # Only allow safe mathematical operations
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return {"error": "Invalid characters in expression"}
            
            result = eval(expression)
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"error": f"Calculation error: {str(e)}"}
    
    @staticmethod
    def convert_units(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
        """Convert between common units."""
        # Basic unit conversions
        conversions = {
            "length": {
                "m_to_ft": 3.28084,
                "ft_to_m": 0.3048,
                "km_to_mi": 0.621371,
                "mi_to_km": 1.60934
            },
            "weight": {
                "kg_to_lb": 2.20462,
                "lb_to_kg": 0.453592
            },
            "temperature": {
                "c_to_f": lambda c: c * 9/5 + 32,
                "f_to_c": lambda f: (f - 32) * 5/9
            }
        }
        
        try:
            # This is a simplified version - you could expand this significantly
            if from_unit == "c" and to_unit == "f":
                result = conversions["temperature"]["c_to_f"](value)
            elif from_unit == "f" and to_unit == "c":
                result = conversions["temperature"]["f_to_c"](value)
            else:
                return {"error": f"Conversion from {from_unit} to {to_unit} not supported"}
            
            return {
                "value": value,
                "from_unit": from_unit,
                "to_unit": to_unit,
                "result": round(result, 2)
            }
        except Exception as e:
            return {"error": f"Conversion error: {str(e)}"}

class ToolManager:
    """Manager for all available tools."""
    
    def __init__(self):
        self.system_tools = SystemTools()
        self.file_tools = FileTools()
        self.web_tools = WebTools()
        self.command_tools = CommandTools()
        self.utility_tools = UtilityTools()
    
    def get_available_tools(self) -> Dict[str, Any]:
        """Get list of all available tools."""
        return {
            "system": {
                "get_system_info": "Get comprehensive system information",
                "get_process_info": "Get information about running processes",
                "get_network_info": "Get network information"
            },
            "file": {
                "list_directory": "List contents of a directory",
                "read_file": "Read a file's contents",
                "search_files": "Search for files matching a pattern"
            },
            "web": {
                "check_url": "Check if a URL is accessible",
                "get_weather": "Get weather information for a city"
            },
            "command": {
                "execute_command": "Execute a system command safely"
            },
            "utility": {
                "get_current_time": "Get current timestamp",
                "calculate": "Safely evaluate a mathematical expression",
                "convert_units": "Convert between common units"
            }
        }
    
    def execute_tool(self, tool_category: str, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific tool."""
        try:
            if tool_category == "system":
                tool_obj = self.system_tools
            elif tool_category == "file":
                tool_obj = self.file_tools
            elif tool_category == "web":
                tool_obj = self.web_tools
            elif tool_category == "command":
                tool_obj = self.command_tools
            elif tool_category == "utility":
                tool_obj = self.utility_tools
            else:
                return {"error": f"Unknown tool category: {tool_category}"}
            
            if not hasattr(tool_obj, tool_name):
                return {"error": f"Tool {tool_name} not found in category {tool_category}"}
            
            method = getattr(tool_obj, tool_name)
            result = method(**kwargs)
            
            return {
                "tool_category": tool_category,
                "tool_name": tool_name,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_category}.{tool_name}: {e}")
            return {"error": str(e)}

# Global tool manager instance
tool_manager = ToolManager()

def get_tool_manager() -> ToolManager:
    """Get the global tool manager instance."""
    return tool_manager 