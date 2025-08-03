#!/usr/bin/env python3
"""
Clean up binary files that are causing UTF-8 encoding errors
"""

import os
import glob
from pathlib import Path

def cleanup_binary_files():
    """Remove or fix binary files that are causing syntax errors."""
    
    # Files that are causing UTF-8 encoding errors (these are binary files)
    binary_files = [
        "realistic_malware.py",
        "test_malware.py",
        "simple_final_test/crypto_miner.py",
        "simple_final_test/system_backdoor.py", 
        "simple_final_test/file_encryptor.py",
        "comprehensive_test/complex_ransomware.py",
        "comprehensive_test/cryptominer_real_2.py",
        "comprehensive_test/advanced_cryptominer.py",
        "comprehensive_test/simple_ransomware.py",
        "comprehensive_test/ransomware_real_0.py",
        "comprehensive_test/trojan_real_1.py",
        "comprehensive_test/stealth_trojan.py",
        "four_way_test/data_stealer.py",
        "four_way_test/crypto_miner.py",
        "four_way_test/ransomware_encryptor.py",
        "four_way_test/stealth_backdoor.py",
        "real_test/cryptominer_real_2.py",
        "real_test/ransomware_real_0.py",
        "real_test/trojan_real_1.py"
    ]
    
    print("🧹 Cleaning up binary files...")
    
    for file_path in binary_files:
        if os.path.exists(file_path):
            try:
                # Try to read as text first
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # If we can read it as text, it's not a binary file
                    print(f"✅ {file_path}: Valid text file")
            except UnicodeDecodeError:
                # This is a binary file, remove it
                os.remove(file_path)
                print(f"🗑️  {file_path}: Removed binary file")
            except Exception as e:
                print(f"❌ {file_path}: Error - {e}")
    
    # Also clean up any .pyc files
    pyc_files = glob.glob("**/*.pyc", recursive=True)
    for pyc_file in pyc_files:
        try:
            os.remove(pyc_file)
            print(f"🗑️  {pyc_file}: Removed .pyc file")
        except Exception as e:
            print(f"❌ {pyc_file}: Error removing - {e}")
    
    print("✅ Cleanup completed!")

if __name__ == "__main__":
    cleanup_binary_files()