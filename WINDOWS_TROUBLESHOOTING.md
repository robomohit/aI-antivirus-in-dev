# Windows Troubleshooting Guide

## Common Issues and Solutions

### 1. ModuleNotFoundError: No module named 'pandas'

**Problem**: Dependencies not installed in virtual environment

**Solution**:
```batch
# Run the fixed setup script
setup_windows_fixed.bat

# Or manually activate and install:
call venv\Scripts\activate.bat
pip install -r requirements_windows.txt
```

### 2. Virtual Environment Not Activated

**Problem**: Running Python without activating virtual environment

**Solution**:
```batch
# Always activate before running:
call venv\Scripts\activate.bat
python ai_antivirus_windows.py --smart-scan
```

### 3. UnicodeEncodeError

**Problem**: Windows encoding issues with emoji characters

**Solution**: 
- Use the Windows-specific scripts (already fixed)
- All emoji characters have been removed from training scripts

### 4. Permission Errors

**Problem**: Windows security blocking file operations

**Solution**:
- Run Command Prompt as Administrator
- Add antivirus directory to Windows Defender exclusions
- Ensure you have write permissions to the project folder

### 5. Python Not Found

**Problem**: Python not in PATH

**Solution**:
```batch
# Check Python installation
python --version

# If not found, install Python 3.8+ from python.org
# Make sure to check "Add Python to PATH" during installation
```

## Quick Fix Commands

### Complete Reset and Setup:
```batch
# 1. Delete existing venv
rmdir /s /q venv

# 2. Run fixed setup
setup_windows_fixed.bat

# 3. Test installation
activate_windows.bat --smart-scan
```

### Manual Installation:
```batch
# 1. Create virtual environment
python -m venv venv

# 2. Activate
call venv\Scripts\activate.bat

# 3. Install dependencies
pip install -r requirements_windows.txt

# 4. Create dataset and train model
python create_dataset_windows.py
python train_enhanced_model_windows.py

# 5. Run antivirus
python ai_antivirus_windows.py --smart-scan
```

## File Structure Check

Ensure these files exist:
- `ai_antivirus_windows.py`
- `setup_windows_fixed.bat`
- `activate_windows.bat`
- `requirements_windows.txt`
- `create_dataset_windows.py`
- `train_enhanced_model_windows.py`

## Directory Structure

```
ai-antivirus/
├── venv/                    # Virtual environment
├── model/                   # AI models
├── logs/                    # Log files
├── quarantine/              # Quarantined files
├── test_files/              # Test files
├── ai_antivirus_windows.py  # Main antivirus
├── setup_windows_fixed.bat  # Setup script
└── activate_windows.bat     # Activation script
```

## Testing Installation

After setup, test with:
```batch
activate_windows.bat --smart-scan
```

This should run without any ModuleNotFoundError.

## Support

If issues persist:
1. Check Python version (3.8+ required)
2. Ensure virtual environment is activated
3. Verify all dependencies are installed
4. Check Windows Defender exclusions
5. Run as Administrator if needed