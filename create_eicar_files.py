#!/usr/bin/env python3
"""
Create EICAR test files for antivirus testing
Based on official EICAR specifications
"""

import os
from pathlib import Path

# EICAR test string (official)
EICAR_STRING = r'X5O!P%@AP[4\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*'

# EICAR file variations
EICAR_FILES = {
    'eicar.com': EICAR_STRING,
    'eicar.txt': EICAR_STRING,
    'eicar.scr': EICAR_STRING,
    'eicar.bat': f'@echo off\necho {EICAR_STRING}',
    'eicar.ps1': f'Write-Output "{EICAR_STRING}"',
    'eicar.vbs': f'MsgBox "{EICAR_STRING}"',
    'eicar.js': f'alert("{EICAR_STRING}");',
    'eicar.html': f'<html><body><script>alert("{EICAR_STRING}");</script></body></html>',
    'eicar.zip': EICAR_STRING,  # Will be compressed
    'eicar.rar': EICAR_STRING,  # Will be compressed
    'eicar.tar.gz': EICAR_STRING,  # Will be compressed
    'eicar.exe': EICAR_STRING,
    'eicar.dll': EICAR_STRING,
    'eicar.sys': EICAR_STRING,
    'eicar.inf': f'[autorun]\nopen={EICAR_STRING}',
    'eicar.reg': f'Windows Registry Editor Version 5.00\n\n[HKEY_LOCAL_MACHINE\\SOFTWARE\\EICAR]\n"Test"="{EICAR_STRING}"',
    'eicar.ini': f'[EICAR]\nTest={EICAR_STRING}',
    'eicar.cmd': f'@echo off\necho {EICAR_STRING}',
    'eicar.wsf': f'<job id="EICAR"><script language="VBScript">MsgBox "{EICAR_STRING}"</script></job>',
    'eicar.hta': f'<html><head><title>EICAR Test</title></head><body><script>alert("{EICAR_STRING}");</script></body></html>'
}

def create_eicar_files():
    """Create all EICAR test files."""
    eicar_dir = Path("test_files/eicar")
    eicar_dir.mkdir(exist_ok=True)
    
    print("🔐 Creating EICAR test files...")
    
    for filename, content in EICAR_FILES.items():
        file_path = eicar_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ Created: {filename}")
    
    print(f"🎯 Created {len(EICAR_FILES)} EICAR test files")
    return eicar_dir

if __name__ == "__main__":
    create_eicar_files()