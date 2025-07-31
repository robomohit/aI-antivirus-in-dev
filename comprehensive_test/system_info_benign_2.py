
import platform
import psutil
import os

def get_system_info():
    print("=== System Information ===")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python Version: {platform.python_version()}")
    
    # CPU Info
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    print(f"CPU Usage: {cpu_percent}%")
    print(f"CPU Cores: {cpu_count}")
    
    # Memory Info
    memory = psutil.virtual_memory()
    print(f"Total Memory: {memory.total / (1024**3):.2f} GB")
    print(f"Available Memory: {memory.available / (1024**3):.2f} GB")
    print(f"Memory Usage: {memory.percent}%")
    
    # Disk Info
    disk = psutil.disk_usage('/')
    print(f"Total Disk: {disk.total / (1024**3):.2f} GB")
    print(f"Free Disk: {disk.free / (1024**3):.2f} GB")
    print(f"Disk Usage: {disk.percent}%")
    
    # Network Info
    network = psutil.net_io_counters()
    print(f"Bytes Sent: {network.bytes_sent}")
    print(f"Bytes Received: {network.bytes_recv}")
    
if __name__ == "__main__":
    get_system_info()
