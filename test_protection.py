#!/usr/bin/env python3
"""
Test script to verify project file protection
"""
import sys
from pathlib import Path

# Add the current directory to the path so we can import the antivirus
sys.path.append('.')

from ai_antivirus import UltimateAIAntivirus

def test_protection():
    """Test that project files are protected from quarantine."""
    print("🔒 Testing Project File Protection...")
    
    # Create antivirus instance
    antivirus = UltimateAIAntivirus(monitor_path=".", quarantine_enabled=False, scan_mode="normal")
    
    # Test files that should be protected
    protected_files = [
        'ai_antivirus.py',
        'config.py',
        'utils.py',
        'signatures.py',
        'gui.py',
        'test_suite.py',
        'train_deep_model.py',
        'explain_deep_model.py',
        'eda_analysis.py',
        'final_test.py',
        'test_model_integration.py',
        'create_enhanced_dataset.py',
        'setup_windows.bat',
        'run_antivirus.bat',
        'PROJECT_SUMMARY.md'
    ]
    
    print("\nTesting protected files:")
    for file_name in protected_files:
        file_path = Path(file_name)
        if file_path.exists():
            # Test should_scan_file
            should_scan = antivirus._should_scan_file(file_path)
            # Test analyze_file
            analysis = antivirus.analyze_file(file_path)
            
            status = "✅ PROTECTED" if not should_scan and analysis is None else "❌ NOT PROTECTED"
            print(f"  {file_name}: {status}")
        else:
            print(f"  {file_name}: ⚠️ FILE NOT FOUND")
    
    # Test files that should be scanned (malicious test files)
    test_files = [
        'suspicious_malware.bat',
        'test_malware.bat',
        'trojan_test.bat',
        'real_malware_test.bat',
        'ransomware_test.bat'
    ]
    
    print("\nTesting files that should be scanned:")
    for file_name in test_files:
        file_path = Path(file_name)
        if file_path.exists():
            should_scan = antivirus._should_scan_file(file_path)
            status = "✅ SCANNABLE" if should_scan else "❌ PROTECTED (should be scannable)"
            print(f"  {file_name}: {status}")
        else:
            print(f"  {file_name}: ⚠️ FILE NOT FOUND")
    
    print("\n🎯 Protection Test Complete!")

if __name__ == "__main__":
    test_protection()