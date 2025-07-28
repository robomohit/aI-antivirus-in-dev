#!/usr/bin/env python3
"""
Simple Antivirus in Python
A basic file monitoring system that detects suspicious files and logs them.
"""

import os
import sys
import time
import shutil
import logging
from datetime import datetime
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import signal
import argparse


class SimpleAntivirus:
    def __init__(self, monitor_path, quarantine_enabled=True):
        """
        Initialize the antivirus system.
        
        Args:
            monitor_path (str): Path to monitor for suspicious files
            quarantine_enabled (bool): Whether to move suspicious files to quarantine
        """
        self.monitor_path = Path(monitor_path).resolve()
        self.quarantine_enabled = quarantine_enabled
        
        # Create necessary directories
        self.logs_dir = Path("logs")
        self.quarantine_dir = Path("quarantine")
        self._create_directories()
        
        # Setup logging
        self._setup_logging()
        
        # Suspicious file extensions
        self.suspicious_extensions = {
            '.exe', '.bat', '.vbs', '.scr', '.ps1', '.cmd', '.com', 
            '.pif', '.reg', '.js', '.jar', '.msi', '.dll', '.sys'
        }
        
        # Initialize watchdog observer
        self.observer = Observer()
        self.event_handler = AntivirusEventHandler(self)
        
        self.logger.info(f"🚀 Simple Antivirus started")
        self.logger.info(f"📁 Monitoring: {self.monitor_path}")
        self.logger.info(f"🛡️ Quarantine enabled: {self.quarantine_enabled}")
        self.logger.info(f"⚠️ Suspicious extensions: {', '.join(sorted(self.suspicious_extensions))}")
    
    def _create_directories(self):
        """Create logs and quarantine directories if they don't exist."""
        self.logs_dir.mkdir(exist_ok=True)
        if self.quarantine_enabled:
            self.quarantine_dir.mkdir(exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.logs_dir / f"antivirus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def is_suspicious_file(self, file_path):
        """
        Check if a file is suspicious based on its extension.
        
        Args:
            file_path (Path): Path to the file to check
            
        Returns:
            bool: True if file is suspicious, False otherwise
        """
        return file_path.suffix.lower() in self.suspicious_extensions
    
    def quarantine_file(self, file_path):
        """
        Move a suspicious file to the quarantine folder.
        
        Args:
            file_path (Path): Path to the file to quarantine
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.quarantine_enabled:
                return False
            
            # Create quarantine filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            quarantine_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            quarantine_path = self.quarantine_dir / quarantine_name
            
            # Move file to quarantine
            shutil.move(str(file_path), str(quarantine_path))
            self.logger.info(f"🚫 Quarantined: {file_path} -> {quarantine_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to quarantine {file_path}: {e}")
            return False
    
    def handle_suspicious_file(self, file_path):
        """
        Handle a detected suspicious file.
        
        Args:
            file_path (Path): Path to the suspicious file
        """
        self.logger.warning(f"🚨 SUSPICIOUS FILE DETECTED: {file_path}")
        self.logger.info(f"📊 File size: {file_path.stat().st_size} bytes")
        self.logger.info(f"🕒 Last modified: {datetime.fromtimestamp(file_path.stat().st_mtime)}")
        
        # Try to quarantine the file
        if self.quarantine_enabled:
            self.quarantine_file(file_path)
        else:
            self.logger.info(f"⚠️ File left in place (quarantine disabled)")
    
    def scan_directory(self):
        """Scan the monitored directory for existing suspicious files."""
        self.logger.info(f"🔍 Scanning directory: {self.monitor_path}")
        
        suspicious_count = 0
        for file_path in self.monitor_path.rglob("*"):
            if file_path.is_file() and self.is_suspicious_file(file_path):
                suspicious_count += 1
                self.handle_suspicious_file(file_path)
        
        self.logger.info(f"✅ Initial scan complete. Found {suspicious_count} suspicious files.")
    
    def start_monitoring(self):
        """Start monitoring the directory for new files."""
        try:
            # Schedule the observer
            self.observer.schedule(
                self.event_handler, 
                str(self.monitor_path), 
                recursive=True
            )
            self.observer.start()
            
            self.logger.info("👁️ File monitoring started")
            self.logger.info("Press Ctrl+C to stop monitoring")
            
            # Keep the script running
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.stop_monitoring()
        except Exception as e:
            self.logger.error(f"❌ Error during monitoring: {e}")
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop the file monitoring."""
        self.observer.stop()
        self.observer.join()
        self.logger.info("🛑 Antivirus monitoring stopped")


class AntivirusEventHandler(FileSystemEventHandler):
    """Event handler for file system events."""
    
    def __init__(self, antivirus):
        self.antivirus = antivirus
    
    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if self.antivirus.is_suspicious_file(file_path):
            # Small delay to ensure file is fully written
            time.sleep(0.5)
            if file_path.exists():  # Check if file still exists
                self.antivirus.handle_suspicious_file(file_path)
    
    def on_moved(self, event):
        """Handle file move events."""
        if event.is_directory:
            return
        
        file_path = Path(event.dest_path)
        if self.antivirus.is_suspicious_file(file_path):
            self.antivirus.handle_suspicious_file(file_path)


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\n🛑 Received interrupt signal. Shutting down...")
    sys.exit(0)


def main():
    """Main function to run the antivirus."""
    parser = argparse.ArgumentParser(description="Simple Antivirus in Python")
    parser.add_argument(
        "--path", 
        default="Downloads", 
        help="Path to monitor (default: Downloads)"
    )
    parser.add_argument(
        "--no-quarantine", 
        action="store_true", 
        help="Disable quarantine functionality"
    )
    parser.add_argument(
        "--scan-only", 
        action="store_true", 
        help="Only scan existing files, don't monitor for new ones"
    )
    
    args = parser.parse_args()
    
    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize antivirus
    antivirus = SimpleAntivirus(
        monitor_path=args.path,
        quarantine_enabled=not args.no_quarantine
    )
    
    # Perform initial scan
    antivirus.scan_directory()
    
    # Start monitoring if not scan-only mode
    if not args.scan_only:
        antivirus.start_monitoring()
    else:
        print("✅ Scan complete. Exiting...")


if __name__ == "__main__":
    main()