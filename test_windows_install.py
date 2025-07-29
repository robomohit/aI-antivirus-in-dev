#!/usr/bin/env python3
"""
Test script for Windows installation
"""
import sys
import importlib

def test_imports():
    """Test all required imports."""
    print("Testing Windows Installation...")
    print("=" * 40)
    
    required_modules = [
        'pandas',
        'numpy', 
        'torch',
        'sklearn',
        'watchdog',
        'colorama',
        'rich',
        'matplotlib',
        'seaborn',
        'shap'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    print("\n" + "=" * 40)
    
    if failed_imports:
        print(f"❌ {len(failed_imports)} modules failed to import:")
        for module in failed_imports:
            print(f"  - {module}")
        print("\nTo fix, run:")
        print("  pip install -r requirements_windows.txt")
        return False
    else:
        print("✅ All modules imported successfully!")
        print("✅ Windows installation is working correctly!")
        return True

def test_antivirus_files():
    """Test if antivirus files exist."""
    print("\nTesting antivirus files...")
    print("=" * 40)
    
    required_files = [
        'ai_antivirus_windows.py',
        'config.py',
        'utils.py',
        'signatures.py',
        'create_dataset_windows.py',
        'train_enhanced_model_windows.py'
    ]
    
    missing_files = []
    
    for file in required_files:
        try:
            with open(file, 'r') as f:
                print(f"✅ {file}")
        except FileNotFoundError:
            print(f"❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ {len(missing_files)} files missing!")
        return False
    else:
        print("\n✅ All antivirus files found!")
        return True

if __name__ == "__main__":
    print("Windows Installation Test")
    print("=" * 40)
    
    imports_ok = test_imports()
    files_ok = test_antivirus_files()
    
    if imports_ok and files_ok:
        print("\n🎉 Windows installation is ready!")
        print("You can now run:")
        print("  python ai_antivirus_windows.py --smart-scan")
    else:
        print("\n💥 Windows installation needs fixing!")
        print("Please run setup_windows_fixed.bat")