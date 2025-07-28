#!/usr/bin/env python3
"""
Setup script for Simple Antivirus
Installs dependencies and runs initial tests.
"""

import subprocess
import sys
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        # Try to install watchdog
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "watchdog", "--break-system-packages"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        print("💡 Try running: pip install watchdog --break-system-packages")
        return False


def test_antivirus():
    """Run a quick test of the antivirus."""
    print("🧪 Testing antivirus functionality...")
    
    # Create test directory
    test_dir = Path("setup_test")
    test_dir.mkdir(exist_ok=True)
    
    # Create a test suspicious file
    test_file = test_dir / "test.exe"
    with open(test_file, 'w') as f:
        f.write("This is a test file")
    
    try:
        # Run antivirus scan
        result = subprocess.run([
            sys.executable, "simple_antivirus.py", 
            "--path", str(test_dir), "--scan-only"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Antivirus test passed")
            return True
        else:
            print("❌ Antivirus test failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False
    finally:
        # Clean up
        import shutil
        try:
            shutil.rmtree(test_dir)
        except:
            pass


def main():
    """Main setup function."""
    print("🛡️ Simple Antivirus Setup")
    print("=" * 30)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install dependencies
    if not install_dependencies():
        return
    
    # Test antivirus
    if not test_antivirus():
        return
    
    print("\n🎉 Setup complete!")
    print("\n📖 Usage:")
    print("  python3 simple_antivirus.py                    # Monitor Downloads folder")
    print("  python3 simple_antivirus.py --path /path/to/monitor")
    print("  python3 simple_antivirus.py --no-quarantine    # Only log, don't quarantine")
    print("  python3 simple_antivirus.py --scan-only        # Scan existing files only")
    print("\n🧪 Run demo:")
    print("  python3 demo.py")
    print("\n📚 Read documentation:")
    print("  cat README.md")


if __name__ == "__main__":
    main()