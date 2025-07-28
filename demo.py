#!/usr/bin/env python3
"""
Demonstration script for the Simple Antivirus
Shows real-time monitoring in action.
"""

import os
import time
import subprocess
import sys
import threading
from pathlib import Path


def create_suspicious_file(test_dir, filename):
    """Create a suspicious file to trigger detection."""
    file_path = test_dir / filename
    with open(file_path, 'w') as f:
        f.write(f"This is a suspicious {filename} file created at {time.strftime('%H:%M:%S')}")
    print(f"📄 Created: {filename}")


def demo_real_time_monitoring():
    """Demonstrate real-time file monitoring."""
    print("🛡️ Simple Antivirus - Real-time Monitoring Demo")
    print("=" * 50)
    
    # Create test directory
    test_dir = Path("demo_downloads")
    test_dir.mkdir(exist_ok=True)
    
    print(f"📁 Monitoring directory: {test_dir}")
    print("⏰ The antivirus will start monitoring in 3 seconds...")
    print("📝 We'll create suspicious files to trigger detection")
    print("🛑 Press Ctrl+C to stop the demo\n")
    
    time.sleep(3)
    
    # Start antivirus in background
    cmd = [sys.executable, "simple_antivirus.py", "--path", str(test_dir)]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    try:
        # Wait for antivirus to start
        time.sleep(2)
        
        # Create suspicious files with delays
        suspicious_files = [
            "malware.exe",
            "script.bat", 
            "virus.vbs",
            "dangerous.ps1",
            "trojan.scr"
        ]
        
        for i, filename in enumerate(suspicious_files):
            print(f"\n⏳ Creating {filename} in {3-i} seconds...")
            time.sleep(3-i)
            
            create_suspicious_file(test_dir, filename)
            
            # Wait for detection
            time.sleep(2)
        
        print(f"\n✅ Demo complete! Check the logs and quarantine folder.")
        print(f"📊 Logs: logs/")
        print(f"🚫 Quarantine: quarantine/")
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    
    finally:
        # Stop the antivirus
        process.terminate()
        process.wait()
        
        # Clean up test directory
        import shutil
        try:
            shutil.rmtree(test_dir)
            print("🧹 Demo files cleaned up")
        except:
            pass


def main():
    """Main demo function."""
    print("🎯 Simple Antivirus Demo")
    print("Choose a demo:")
    print("1. Real-time monitoring demo")
    print("2. Exit")
    
    choice = input("\nEnter your choice (1-2): ").strip()
    
    if choice == "1":
        demo_real_time_monitoring()
    elif choice == "2":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()