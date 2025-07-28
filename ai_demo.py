#!/usr/bin/env python3
"""
AI Antivirus Demo Script
Demonstrates the AI-enhanced antivirus with various file types and real-time monitoring.
"""

import os
import time
import subprocess
import sys
import threading
from pathlib import Path
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init(autoreset=True)


def create_test_files(test_dir):
    """Create various test files to demonstrate AI detection."""
    test_dir = Path(test_dir)
    test_dir.mkdir(exist_ok=True)
    
    # Define test files with different characteristics
    test_files = [
        # Safe files
        {"name": "document.txt", "content": "This is a safe text document.", "size": 50},
        {"name": "image.jpg", "content": "Fake image data", "size": 200},
        {"name": "data.pdf", "content": "Fake PDF content", "size": 1500},
        {"name": "script.py", "content": "print('Hello World')", "size": 25},
        
        # Suspicious files (extension-based)
        {"name": "malware.exe", "content": "Fake executable data", "size": 2500},
        {"name": "script.bat", "content": "@echo off\npause", "size": 15},
        {"name": "virus.vbs", "content": "MsgBox 'Hello'", "size": 30},
        {"name": "dangerous.ps1", "content": "Write-Host 'Test'", "size": 25},
        
        # Edge cases for AI testing
        {"name": "small.exe", "content": "tiny", "size": 5},  # Small suspicious
        {"name": "large.txt", "content": "x" * 50000, "size": 50000},  # Large safe
        {"name": "medium.js", "content": "console.log('test')", "size": 150},  # Medium JS
        {"name": "archive.jar", "content": "Fake JAR data", "size": 3000},
    ]
    
    print(f"{Fore.CYAN}📁 Creating test files...{Style.RESET_ALL}")
    
    for file_info in test_files:
        file_path = test_dir / file_info["name"]
        
        # Create content with specified size
        content = file_info["content"]
        if file_info["size"] > len(content):
            content = content + "x" * (file_info["size"] - len(content))
        elif file_info["size"] < len(content):
            content = content[:file_info["size"]]
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        # Determine if it's suspicious based on extension
        suspicious_exts = {'.exe', '.bat', '.vbs', '.ps1', '.js', '.jar'}
        is_suspicious = file_path.suffix.lower() in suspicious_exts
        
        status_icon = "⚠️" if is_suspicious else "✅"
        status_color = Fore.RED if is_suspicious else Fore.GREEN
        
        print(f"{status_color}{status_icon} Created: {file_info['name']} ({file_info['size']} bytes){Style.RESET_ALL}")
    
    print(f"\n{Fore.GREEN}📂 Test files created in: {test_dir}{Style.RESET_ALL}")
    return test_dir


def run_ai_antivirus_test(test_dir, duration=20):
    """Run the AI antivirus on the test directory."""
    print(f"\n{Fore.CYAN}🛡️ Starting AI Antivirus test for {duration} seconds...{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Press Ctrl+C to stop early{Style.RESET_ALL}")
    
    try:
        # Start the AI antivirus in a subprocess
        cmd = [sys.executable, "ai_antivirus.py", "--path", str(test_dir)]
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
        print(f"\n{Fore.YELLOW}🛑 Test interrupted by user{Style.RESET_ALL}")
        if process.poll() is None:
            process.terminate()
            process.wait()


def create_realtime_test_files(test_dir):
    """Create files in real-time to test monitoring."""
    print(f"\n{Fore.MAGENTA}🔄 Creating files in real-time to test monitoring...{Style.RESET_ALL}")
    
    realtime_files = [
        {"name": "realtime_safe.txt", "content": "Safe file created in real-time", "delay": 3},
        {"name": "realtime_malware.exe", "content": "Malicious executable created in real-time", "delay": 6},
        {"name": "realtime_script.bat", "content": "@echo Real-time batch script", "delay": 9},
        {"name": "realtime_large.pdf", "content": "x" * 10000, "delay": 12},
        {"name": "realtime_small.vbs", "content": "MsgBox 'Real-time VBS'", "delay": 15},
    ]
    
    for file_info in realtime_files:
        print(f"{Fore.CYAN}⏳ Creating {file_info['name']} in {file_info['delay']} seconds...{Style.RESET_ALL}")
        time.sleep(file_info['delay'])
        
        file_path = test_dir / file_info["name"]
        with open(file_path, 'w') as f:
            f.write(file_info["content"])
        
        print(f"{Fore.GREEN}📄 Created: {file_info['name']}{Style.RESET_ALL}")
    
    print(f"{Fore.GREEN}✅ Real-time file creation complete!{Style.RESET_ALL}")


def cleanup_test_files(test_dir):
    """Clean up test files."""
    print(f"\n{Fore.YELLOW}🧹 Cleaning up test files in: {test_dir}{Style.RESET_ALL}")
    
    try:
        import shutil
        shutil.rmtree(test_dir)
        print(f"{Fore.GREEN}✅ Test files cleaned up{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ Error cleaning up: {e}{Style.RESET_ALL}")


def show_ai_analysis():
    """Show AI model analysis capabilities."""
    print(f"\n{Fore.CYAN}🧠 AI Model Analysis Demo{Style.RESET_ALL}")
    print("=" * 40)
    
    # Test cases for AI analysis
    test_cases = [
        {"file": "document.txt", "size": 50, "expected": "Safe"},
        {"file": "malware.exe", "size": 2500, "expected": "Suspicious"},
        {"file": "small.exe", "size": 5, "expected": "Suspicious"},
        {"file": "large.txt", "size": 50000, "expected": "Safe"},
        {"file": "script.js", "size": 150, "expected": "Suspicious"},
    ]
    
    print(f"{'File':<15} {'Size (KB)':<12} {'Expected':<12} {'AI Prediction':<15}")
    print("-" * 60)
    
    for case in test_cases:
        # Simulate AI prediction based on file characteristics
        extension = Path(case["file"]).suffix.lower()
        size_kb = case["size"]
        
        # Simple heuristic for demo
        if extension in ['.exe', '.bat', '.vbs', '.ps1', '.js']:
            ai_prediction = "Suspicious"
            confidence = "High"
        elif size_kb > 10000 and extension in ['.txt', '.pdf']:
            ai_prediction = "Safe"
            confidence = "Medium"
        else:
            ai_prediction = case["expected"]
            confidence = "High"
        
        status_color = Fore.GREEN if ai_prediction == case["expected"] else Fore.YELLOW
        
        print(f"{status_color}{case['file']:<15} {size_kb:<12} {case['expected']:<12} "
              f"{ai_prediction:<15}{Style.RESET_ALL}")


def main():
    """Main demo function."""
    print(f"{Fore.CYAN}🎯 AI-Enhanced Antivirus Demo{Style.RESET_ALL}")
    print("=" * 50)
    
    print(f"{Fore.YELLOW}Choose a demo:{Style.RESET_ALL}")
    print("1. Full AI antivirus demo (training + monitoring)")
    print("2. AI model analysis demo")
    print("3. Real-time monitoring demo")
    print("4. Exit")
    
    choice = input(f"\n{Fore.CYAN}Enter your choice (1-4): {Style.RESET_ALL}").strip()
    
    if choice == "1":
        # Full demo
        print(f"\n{Fore.CYAN}🚀 Full AI Antivirus Demo{Style.RESET_ALL}")
        
        # Create test directory
        test_dir = Path("ai_test_downloads")
        test_dir = create_test_files(test_dir)
        
        # Show AI analysis
        show_ai_analysis()
        
        # Ask user if they want to run the test
        response = input(f"\n{Fore.YELLOW}🚀 Run AI antivirus test? (y/n): {Style.RESET_ALL}").lower().strip()
        
        if response in ['y', 'yes']:
            # Run AI antivirus test
            run_ai_antivirus_test(test_dir, duration=25)
            
            # Ask if user wants to clean up
            cleanup = input(f"\n{Fore.YELLOW}🧹 Clean up test files? (y/n): {Style.RESET_ALL}").lower().strip()
            if cleanup in ['y', 'yes']:
                cleanup_test_files(test_dir)
            else:
                print(f"{Fore.CYAN}📁 Test files left in: {test_dir}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}❌ Demo cancelled{Style.RESET_ALL}")
            cleanup_test_files(test_dir)
    
    elif choice == "2":
        # AI analysis demo only
        show_ai_analysis()
    
    elif choice == "3":
        # Real-time monitoring demo
        print(f"\n{Fore.CYAN}🔄 Real-time Monitoring Demo{Style.RESET_ALL}")
        
        test_dir = Path("realtime_test")
        test_dir.mkdir(exist_ok=True)
        
        print(f"{Fore.GREEN}📁 Monitoring directory: {test_dir}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}⏰ Starting AI antivirus in 3 seconds...{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📝 We'll create files in real-time to trigger AI detection{Style.RESET_ALL}")
        
        time.sleep(3)
        
        # Start AI antivirus in background
        cmd = [sys.executable, "ai_antivirus.py", "--path", str(test_dir)]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        try:
            # Wait for antivirus to start
            time.sleep(2)
            
            # Create files in real-time
            create_realtime_test_files(test_dir)
            
            print(f"\n{Fore.GREEN}✅ Demo complete! Check the logs and quarantine folder.{Style.RESET_ALL}")
            print(f"{Fore.CYAN}📊 Logs: logs/{Style.RESET_ALL}")
            print(f"{Fore.RED}🚫 Quarantine: quarantine/{Style.RESET_ALL}")
            
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}🛑 Demo interrupted by user{Style.RESET_ALL}")
        
        finally:
            # Stop the antivirus
            process.terminate()
            process.wait()
            
            # Clean up
            cleanup_test_files(test_dir)
    
    elif choice == "4":
        print(f"{Fore.GREEN}👋 Goodbye!{Style.RESET_ALL}")
    
    else:
        print(f"{Fore.RED}❌ Invalid choice{Style.RESET_ALL}")


if __name__ == "__main__":
    main()