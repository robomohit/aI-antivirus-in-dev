@echo off
echo 🚀 Ultimate AI Antivirus v5.X - Windows Setup
echo ================================================

echo.
echo 📋 Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python 3.8+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo ✅ Python found! Creating virtual environment...
python -m venv venv

echo.
echo 🔧 Activating virtual environment...
call venv\Scripts\activate

echo.
echo 📦 Installing dependencies...
pip install -r requirements.txt

echo.
echo Creating malware dataset...
python create_dataset_windows.py

echo Training AI model (this may take a few minutes)...
python train_enhanced_model.py

echo.
echo 🎉 Setup complete! 
echo.
echo To run the antivirus:
echo    GUI Mode: python ai_antivirus_windows.py --gui
echo    Smart Scan: python ai_antivirus_windows.py --smart-scan
echo    Full Scan: python ai_antivirus_windows.py --full-scan
echo.
echo 📖 See WINDOWS_SETUP_GUIDE.md for detailed instructions
echo.
pause