#!/usr/bin/env python3
"""
Test script for the Simple Antivirus
Creates test files to demonstrate the antivirus functionality.
"""

import os
import time
import subprocess
import sys
from pathlib import Path


def create_test_files(test_dir):
    """Create test files to demonstrate antivirus detection."""
    test_dir = Path(test_dir)
    test_dir.mkdir(exist_ok=True)
    
    # Create some safe files
    safe_files = [
        "document.txt",
        "image.jpg", 
        "data.csv",
        "script.py"
    ]
    
    # Create some suspicious files
    suspicious_files = [
        "test.exe",
        "script.bat",
        "malware.vbs",
        "suspicious.ps1",
        "dangerous.scr"
    ]
    
    print("📁 Creating test files...")
    
    # Create safe files
    for filename in safe_files:
        file_path = test_dir / filename
        with open(file_path, 'w') as f:
            f.write(f"This is a safe {filename} file for testing.")
        print(f"✅ Created safe file: {filename}")
    
    # Create suspicious files
    for filename in suspicious_files:
        file_path = test_dir / filename
        with open(file_path, 'w') as f:
            f.write(f"This is a suspicious {filename} file for testing.")
        print(f"⚠️ Created suspicious file: {filename}")
    
    print(f"\n📂 Test files created in: {test_dir}")
    return test_dir


def run_antivirus_test(test_dir, duration=10):
    """Run the antivirus on the test directory."""
    print(f"\n🛡️ Starting antivirus test for {duration} seconds...")
    print("Press Ctrl+C to stop early")
    
    try:
        # Start the antivirus in a subprocess
        cmd = [sys.executable, "simple_antivirus.py", "--path", str(test_dir)]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Monitor the output
        start_time = time.time()
        while time.time() - start_time < duration:
            output = process.stdout.readline()
            if output:
                print(output.rstrip())
            
            # Check if process is still running
            if process.poll() is not None:
                break
                
            time.sleep(0.1)
        
        # Stop the process
        process.terminate()
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        if process.poll() is None:
            process.terminate()
            process.wait()


def cleanup_test_files(test_dir):
    """Clean up test files."""
    print(f"\n🧹 Cleaning up test files in: {test_dir}")
    
    try:
        import shutil
        shutil.rmtree(test_dir)
        print("✅ Test files cleaned up")
    except Exception as e:
        print(f"❌ Error cleaning up: {e}")


def main():
    """Main test function."""
    print("🧪 Simple Antivirus Test Script")
    print("=" * 40)
    
    # Create test directory
    test_dir = Path("test_downloads")
    
    # Create test files
    test_dir = create_test_files(test_dir)
    
    print(f"\n📋 Test files created:")
    print("Safe files: document.txt, image.jpg, data.csv, script.py")
    print("Suspicious files: test.exe, script.bat, malware.vbs, suspicious.ps1, dangerous.scr")
    
    # Ask user if they want to run the test
    response = input("\n🚀 Run antivirus test? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        # Run antivirus test
        run_antivirus_test(test_dir, duration=15)
        
        # Ask if user wants to clean up
        cleanup = input("\n🧹 Clean up test files? (y/n): ").lower().strip()
        if cleanup in ['y', 'yes']:
            cleanup_test_files(test_dir)
        else:
            print(f"📁 Test files left in: {test_dir}")
    else:
        print("❌ Test cancelled")
        cleanup_test_files(test_dir)


if __name__ == "__main__":
    main()