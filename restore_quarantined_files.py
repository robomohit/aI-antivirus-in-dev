#!/usr/bin/env python3
"""
Restore quarantined legitimate files
"""
import os
import shutil
from pathlib import Path
from datetime import datetime

def restore_quarantined_files():
    """Restore files that were incorrectly quarantined."""
    quarantine_dir = Path("quarantine")
    
    if not quarantine_dir.exists():
        print("❌ Quarantine directory not found")
        return
    
    # List of files that should be restored
    files_to_restore = [
        # Setup files
        "setup_windows.bat",
        "requirements_windows.txt",
        "README_WINDOWS.md",
        
        # Python virtual environment files
        "python.exe", "pythonw.exe", "pip.exe", "pip3.exe",
        "activate.bat", "deactivate.bat", "Activate.ps1",
        "pyvenv.cfg", "pywin32_postinstall.exe",
        
        # Python package DLLs
        "lib_lightgbm.dll", "msvcp140.dll", "vcomp140.dll",
        "pywintypes311.dll", "pythoncom311.dll",
        
        # Other legitimate files
        "*.cfg", "*.ini", "*.json", "*.xml"
    ]
    
    restored_count = 0
    
    print("🔍 Scanning quarantine directory...")
    
    for quarantined_file in quarantine_dir.glob("quarantined_*"):
        file_name = quarantined_file.name.replace("quarantined_", "").split("_", 1)[-1]
        
        # Check if this file should be restored
        should_restore = False
        for pattern in files_to_restore:
            if pattern in file_name or file_name.endswith(pattern.replace("*", "")):
                should_restore = True
                break
        
        # Also restore files from venv directory
        if "venv" in str(quarantined_file) or "env" in str(quarantined_file):
            should_restore = True
        
        if should_restore:
            try:
                # Determine original location
                original_path = None
                
                # Check if it's a venv file
                if "venv" in str(quarantined_file):
                    # Try to restore to venv directory
                    venv_path = Path("venv")
                    if venv_path.exists():
                        original_path = venv_path / file_name
                    else:
                        # Create venv directory if it doesn't exist
                        venv_path.mkdir(exist_ok=True)
                        original_path = venv_path / file_name
                else:
                    # Restore to current directory
                    original_path = Path(file_name)
                
                # Restore the file
                shutil.move(str(quarantined_file), str(original_path))
                print(f"✅ Restored: {file_name}")
                restored_count += 1
                
            except Exception as e:
                print(f"❌ Failed to restore {file_name}: {e}")
    
    print(f"\n📊 Restoration complete!")
    print(f"✅ Files restored: {restored_count}")
    
    if restored_count > 0:
        print("🛡️  Note: The antivirus has been updated to prevent false positives on legitimate files.")
        print("📝 Future scans will exclude virtual environment and Python package files.")

if __name__ == "__main__":
    restore_quarantined_files()