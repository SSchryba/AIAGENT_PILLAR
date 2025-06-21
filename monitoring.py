#!/usr/bin/env python3
"""
Performance Monitoring System
Tracks response times, error rates, and system metrics for the AI agent.
"""

import time
import psutil
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock
import json

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_counts: deque = field(default_factory=lambda: deque(maxlen=1000))
    request_counts: deque = field(default_factory=lambda: deque(maxlen=1000))
    cache_hits: deque = field(default_factory=lambda: deque(maxlen=1000))
    memory_usage: deque = field(default_factory=lambda: deque(maxlen=1000))
    cpu_usage: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_response_time(self, duration: float, endpoint: str = "chat"):
        """Add a response time measurement."""
        self.response_times.append(MetricPoint(
            timestamp=time.time(),
            value=duration,
            tags={"endpoint": endpoint}
        ))
    
    def add_error(self, error_type: str, endpoint: str = "chat"):
        """Add an error count."""
        self.error_counts.append(MetricPoint(
            timestamp=time.time(),
            value=1.0,
            tags={"error_type": error_type, "endpoint": endpoint}
        ))
    
    def add_request(self, endpoint: str = "chat"):
        """Add a request count."""
        self.request_counts.append(MetricPoint(
            timestamp=time.time(),
            value=1.0,
            tags={"endpoint": endpoint}
        ))
    
    def add_cache_hit(self, hit: bool):
        """Add a cache hit/miss."""
        self.cache_hits.append(MetricPoint(
            timestamp=time.time(),
            value=1.0 if hit else 0.0,
            tags={"cache_result": "hit" if hit else "miss"}
        ))
    
    def add_system_metrics(self):
        """Add current system metrics."""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.append(MetricPoint(
                timestamp=time.time(),
                value=memory.percent,
                tags={"metric": "memory_percent"}
            ))
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_usage.append(MetricPoint(
                timestamp=time.time(),
                value=cpu_percent,
                tags={"metric": "cpu_percent"}
            ))
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")

class PerformanceMonitor:
    """Main performance monitoring class."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.metrics = PerformanceMetrics()
        self.lock = Lock()
        self.start_time = time.time()
        
        # Configuration
        self.metrics_retention_hours = 24
        self.cleanup_interval_seconds = 3600  # 1 hour
        self.last_cleanup = time.time()
    
    def start_request(self, endpoint: str = "chat") -> str:
        """Start timing a request and return a request ID."""
        if not self.enabled:
            return ""
        
        request_id = f"{endpoint}_{int(time.time() * 1000)}"
        
        with self.lock:
            self.metrics.add_request(endpoint)
        
        return request_id
    
    def end_request(self, request_id: str, duration: float, success: bool = True, error_type: str = None):
        """End timing a request and record metrics."""
        if not self.enabled or not request_id:
            return
        
        endpoint = request_id.split('_')[0] if '_' in request_id else "unknown"
        
        with self.lock:
            self.metrics.add_response_time(duration, endpoint)
            
            if not success and error_type:
                self.metrics.add_error(error_type, endpoint)
    
    def record_cache_hit(self, hit: bool):
        """Record a cache hit or miss."""
        if not self.enabled:
            return
        
        with self.lock:
            self.metrics.add_cache_hit(hit)
    
    def collect_system_metrics(self):
        """Collect current system metrics."""
        if not self.enabled:
            return
        
        with self.lock:
            self.metrics.add_system_metrics()
    
    def cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        if not self.enabled:
            return
        
        current_time = time.time()
        cutoff_time = current_time - (self.metrics_retention_hours * 3600)
        
        with self.lock:
            # Clean up response times
            self.metrics.response_times = deque(
                [m for m in self.metrics.response_times if m.timestamp > cutoff_time],
                maxlen=1000
            )
            
            # Clean up error counts
            self.metrics.error_counts = deque(
                [m for m in self.metrics.error_counts if m.timestamp > cutoff_time],
                maxlen=1000
            )
            
            # Clean up request counts
            self.metrics.request_counts = deque(
                [m for m in self.metrics.request_counts if m.timestamp > cutoff_time],
                maxlen=1000
            )
            
            # Clean up cache hits
            self.metrics.cache_hits = deque(
                [m for m in self.metrics.cache_hits if m.timestamp > cutoff_time],
                maxlen=1000
            )
            
            # Clean up system metrics (keep more recent ones)
            self.metrics.memory_usage = deque(
                [m for m in self.metrics.memory_usage if m.timestamp > cutoff_time],
                maxlen=1000
            )
            
            self.metrics.cpu_usage = deque(
                [m for m in self.metrics.cpu_usage if m.timestamp > cutoff_time],
                maxlen=1000
            )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        if not self.enabled:
            return {"enabled": False}
        
        with self.lock:
            # Check if cleanup is needed
            current_time = time.time()
            if current_time - self.last_cleanup > self.cleanup_interval_seconds:
                self.cleanup_old_metrics()
                self.last_cleanup = current_time
            
            # Calculate response time statistics
            response_times = [m.value for m in self.metrics.response_times]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            max_response_time = max(response_times) if response_times else 0
            min_response_time = min(response_times) if response_times else 0
            
            # Calculate error rate
            total_requests = len(self.metrics.request_counts)
            total_errors = len(self.metrics.error_counts)
            error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
            
            # Calculate cache hit rate
            cache_hits = [m.value for m in self.metrics.cache_hits]
            cache_hit_rate = (sum(cache_hits) / len(cache_hits) * 100) if cache_hits else 0
            
            # Get latest system metrics
            latest_memory = self.metrics.memory_usage[-1].value if self.metrics.memory_usage else 0
            latest_cpu = self.metrics.cpu_usage[-1].value if self.metrics.cpu_usage else 0
            
            return {
                "enabled": True,
                "uptime_seconds": current_time - self.start_time,
                "response_times": {
                    "average_ms": round(avg_response_time * 1000, 2),
                    "max_ms": round(max_response_time * 1000, 2),
                    "min_ms": round(min_response_time * 1000, 2),
                    "total_requests": total_requests
                },
                "errors": {
                    "total_errors": total_errors,
                    "error_rate_percent": round(error_rate, 2)
                },
                "cache": {
                    "hit_rate_percent": round(cache_hit_rate, 2),
                    "total_cache_operations": len(cache_hits)
                },
                "system": {
                    "memory_usage_percent": round(latest_memory, 2),
                    "cpu_usage_percent": round(latest_cpu, 2)
                },
                "metrics_retention_hours": self.metrics_retention_hours
            }
    
    def get_detailed_metrics(self, hours: int = 1) -> Dict[str, Any]:
        """Get detailed metrics for the specified time period."""
        if not self.enabled:
            return {"enabled": False}
        
        cutoff_time = time.time() - (hours * 3600)
        
        with self.lock:
            # Filter metrics by time
            recent_response_times = [
                {"timestamp": m.timestamp, "value": m.value, "tags": m.tags}
                for m in self.metrics.response_times if m.timestamp > cutoff_time
            ]
            
            recent_errors = [
                {"timestamp": m.timestamp, "value": m.value, "tags": m.tags}
                for m in self.metrics.error_counts if m.timestamp > cutoff_time
            ]
            
            recent_requests = [
                {"timestamp": m.timestamp, "value": m.value, "tags": m.tags}
                for m in self.metrics.request_counts if m.timestamp > cutoff_time
            ]
            
            recent_cache_hits = [
                {"timestamp": m.timestamp, "value": m.value, "tags": m.tags}
                for m in self.metrics.cache_hits if m.timestamp > cutoff_time
            ]
            
            recent_system_metrics = {
                "memory": [
                    {"timestamp": m.timestamp, "value": m.value, "tags": m.tags}
                    for m in self.metrics.memory_usage if m.timestamp > cutoff_time
                ],
                "cpu": [
                    {"timestamp": m.timestamp, "value": m.value, "tags": m.tags}
                    for m in self.metrics.cpu_usage if m.timestamp > cutoff_time
                ]
            }
            
            return {
                "enabled": True,
                "time_period_hours": hours,
                "response_times": recent_response_times,
                "errors": recent_errors,
                "requests": recent_requests,
                "cache_hits": recent_cache_hits,
                "system_metrics": recent_system_metrics
            }
    
    def export_metrics(self, filepath: str) -> bool:
        """Export metrics to a JSON file."""
        if not self.enabled:
            return False
        
        try:
            metrics_data = {
                "export_timestamp": datetime.now().isoformat(),
                "summary": self.get_metrics_summary(),
                "detailed": self.get_detailed_metrics(hours=24)
            }
            
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            logger.info(f"Metrics exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False
    
    def reset_metrics(self):
        """Reset all metrics."""
        if not self.enabled:
            return
        
        with self.lock:
            self.metrics = PerformanceMetrics()
            self.start_time = time.time()
            self.last_cleanup = time.time()
        
        logger.info("Performance metrics reset")

# Global monitor instance
_global_monitor = None

def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor(enabled=True)
    return _global_monitor

def start_monitoring():
    """Start the performance monitoring system."""
    monitor = get_monitor()
    logger.info("Performance monitoring started")
    return monitor

def stop_monitoring():
    """Stop the performance monitoring system."""
    global _global_monitor
    if _global_monitor:
        _global_monitor.enabled = False
        _global_monitor = None
        logger.info("Performance monitoring stopped")

# Decorator for automatic monitoring
def monitor_performance(endpoint: str = "unknown"):
    """Decorator to automatically monitor function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_monitor()
            if not monitor.enabled:
                return func(*args, **kwargs)
            
            request_id = monitor.start_request(endpoint)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                monitor.end_request(request_id, time.time() - start_time, success=True)
                return result
            except Exception as e:
                monitor.end_request(
                    request_id, 
                    time.time() - start_time, 
                    success=False, 
                    error_type=type(e).__name__
                )
                raise
        
        return wrapper
    return decorator 