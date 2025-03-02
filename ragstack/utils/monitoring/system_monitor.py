"""
System monitoring for resource usage during document processing.
"""
import threading
import time
import psutil
from typing import Dict, List, Optional, Any, Callable
import platform
import os

from ragstack.utils.logging import setup_logger
from ragstack.utils.monitoring.metrics import metrics_collector

# Initialize logger
logger = setup_logger("ragstack.utils.monitoring.system")

class SystemMonitor:
    """
    Monitors system resource usage during document processing.
    Collects CPU, memory, disk I/O, and other system metrics.
    """
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """Singleton pattern to ensure only one system monitor exists."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SystemMonitor, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the system monitor."""
        with self._lock:
            if self._initialized:
                return
                
            self._monitoring = False
            self._monitor_thread = None
            self._sample_interval = 5  # seconds
            self._callbacks: List[Callable[[Dict[str, float]], None]] = []
            
            # Store system info
            self._system_info = {
                "os": platform.system(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(logical=True),
                "physical_cpu_count": psutil.cpu_count(logical=False),
            }
            
            self._initialized = True
            logger.info("System monitor initialized")
    
    def start_monitoring(self) -> None:
        """Start monitoring system resources."""
        with self._lock:
            if self._monitoring:
                logger.debug("System monitoring already running")
                return
                
            self._monitoring = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_resources,
                daemon=True
            )
            self._monitor_thread.start()
            logger.info("System resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring system resources."""
        with self._lock:
            if not self._monitoring:
                logger.debug("System monitoring not running")
                return
                
            self._monitoring = False
            if self._monitor_thread:
                # Do not join the thread, just set the flag and let it exit naturally
                logger.info("System resource monitoring stopping")
    
    def register_callback(self, callback: Callable[[Dict[str, float]], None]) -> None:
        """
        Register a callback to be called with resource metrics.
        
        Args:
            callback: Function that takes a dictionary of metrics
        """
        with self._lock:
            self._callbacks.append(callback)
    
    def _monitor_resources(self) -> None:
        """Continuously monitor system resources and record metrics."""
        # Initialize with safe defaults
        try:
            last_disk_io = psutil.disk_io_counters()
        except Exception as e:
            logger.warning(f"Unable to get disk I/O counters: {e}")
            last_disk_io = None
            
        try:
            last_net_io = psutil.net_io_counters()
        except Exception as e:
            logger.warning(f"Unable to get network I/O counters: {e}")
            last_net_io = None
            
        last_time = time.time()
        
        while self._monitoring:
            try:
                # Collect system metrics
                current_time = time.time()
                time_delta = current_time - last_time
                
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                per_cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used_mb = memory.used / (1024 * 1024)
                memory_total_mb = memory.total / (1024 * 1024)
                
                # Disk metrics
                disk_read_mbps = 0
                disk_write_mbps = 0
                try:
                    current_disk_io = psutil.disk_io_counters()
                    if last_disk_io and current_disk_io:
                        disk_read_mbps = ((current_disk_io.read_bytes - last_disk_io.read_bytes) 
                                    / time_delta) / (1024 * 1024)
                        disk_write_mbps = ((current_disk_io.write_bytes - last_disk_io.write_bytes) 
                                     / time_delta) / (1024 * 1024)
                        last_disk_io = current_disk_io
                except Exception as e:
                    logger.debug(f"Skipping disk I/O monitoring: {e}")
                    current_disk_io = None
                
                # Network metrics
                net_recv_mbps = 0
                net_sent_mbps = 0
                try:
                    current_net_io = psutil.net_io_counters()
                    if last_net_io and current_net_io:
                        net_recv_mbps = ((current_net_io.bytes_recv - last_net_io.bytes_recv) 
                                   / time_delta) / (1024 * 1024)
                        net_sent_mbps = ((current_net_io.bytes_sent - last_net_io.bytes_sent) 
                                   / time_delta) / (1024 * 1024)
                        last_net_io = current_net_io
                except Exception as e:
                    logger.debug(f"Skipping network I/O monitoring: {e}")
                    current_net_io = None
                
                # Process metrics
                process = psutil.Process()
                process_cpu_percent = process.cpu_percent(interval=None)
                process_memory_mb = process.memory_info().rss / (1024 * 1024)
                
                # Compile metrics
                metrics = {
                    "cpu.total_percent": cpu_percent,
                    "memory.percent": memory_percent,
                    "memory.used_mb": memory_used_mb,
                    "memory.total_mb": memory_total_mb,
                    "disk.read_mbps": disk_read_mbps,
                    "disk.write_mbps": disk_write_mbps,
                    "network.recv_mbps": net_recv_mbps,
                    "network.sent_mbps": net_sent_mbps,
                    "process.cpu_percent": process_cpu_percent,
                    "process.memory_mb": process_memory_mb,
                }
                
                # Add per-CPU metrics
                for i, cpu_percent in enumerate(per_cpu_percent):
                    metrics[f"cpu.core{i}_percent"] = cpu_percent
                
                # Record metrics
                for key, value in metrics.items():
                    metrics_collector.record_system_metric(key, value)
                
                # Call registered callbacks
                callbacks = list(self._callbacks)  # Make a copy to avoid lock issues
                for callback in callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"Error in system monitor callback: {e}")
                
                # Update time for next iteration
                # (disk_io and net_io are updated above)
                last_time = current_time
                
                # Sleep for the interval, but check if we should stop more frequently
                for _ in range(int(self._sample_interval * 2)):
                    if not self._monitoring:
                        break
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error monitoring system resources: {e}")
                # Sleep briefly to avoid tight loop on persistent errors
                time.sleep(1)
                
        logger.info("System resource monitoring stopped")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get basic system information.
        
        Returns:
            Dictionary of system information
        """
        return self._system_info.copy()
    
    def set_sample_interval(self, interval: float) -> None:
        """
        Set the monitoring sample interval.
        
        Args:
            interval: Sample interval in seconds
        """
        with self._lock:
            self._sample_interval = max(1.0, interval)  # Minimum 1 second
            logger.debug(f"System monitor sample interval set to {self._sample_interval}s")

# Singleton instance
system_monitor = SystemMonitor()