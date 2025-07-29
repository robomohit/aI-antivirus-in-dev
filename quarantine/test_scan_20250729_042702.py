#!/usr/bin/env python3
"""
Quick test script to verify scanning functionality
"""

import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.append('.')

try:
    from ai_antivirus_windows import UltimateAIAntivirus
    print("✅ Successfully imported antivirus")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_scan():
    """Test the scanning functionality."""
    print("🧪 Testing antivirus scan functionality...")
    
    # Create antivirus instance
    try:
        antivirus = UltimateAIAntivirus('.', scan_mode="normal")
        print("✅ Antivirus initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing antivirus: {e}")
        return False
    
    # Test file discovery
    print("🔍 Testing file discovery...")
    try:
        files = antivirus._get_files_to_scan()
        print(f"✅ Found {len(files)} files to scan")
        
        if len(files) > 0:
            print(f"📁 Sample files:")
            for i, file in enumerate(files[:5]):
                print(f"   {i+1}. {file}")
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more files")
        else:
            print("⚠️ No files found to scan")
            
    except Exception as e:
        print(f"❌ Error in file discovery: {e}")
        return False
    
    # Test scanning
    print("🔍 Testing scan functionality...")
    try:
        start_time = time.time()
        antivirus.scan_directory(show_progress=True)
        scan_time = time.time() - start_time
        
        print(f"✅ Scan completed in {scan_time:.2f} seconds")
        print(f"📊 Statistics:")
        print(f"   Files scanned: {antivirus.stats['files_scanned']}")
        print(f"   Threats found: {antivirus.stats['threats_found']}")
        print(f"   AI detections: {antivirus.stats['ai_detections']}")
        print(f"   Extension detections: {antivirus.stats['extension_detections']}")
        print(f"   Both detections: {antivirus.stats['both_detections']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during scan: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Ultimate AI Antivirus - Scan Test")
    print("=" * 50)
    
    success = test_scan()
    
    if success:
        print("\n✅ All tests passed! Antivirus is working correctly.")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
    
    print("\n" + "=" * 50)