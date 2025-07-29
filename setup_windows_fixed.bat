@echo off
echo ================================================
echo Ultimate AI Antivirus v5.X - Windows Setup
echo ================================================

echo.
echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo.
echo Python found! Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment!
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing dependencies...
pip install torch>=2.0.0
pip install scikit-learn>=1.3.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install watchdog>=3.0.0
pip install colorama>=0.4.6
pip install rich>=13.0.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install shap>=0.41.0

echo.
echo Creating necessary directories...
if not exist "model" mkdir model
if not exist "logs" mkdir logs
if not exist "quarantine" mkdir quarantine
if not exist "test_files" mkdir test_files

echo.
echo Creating dataset...
python create_dataset_windows.py

echo.
echo Training AI model (this may take a few minutes)...
python train_enhanced_model_windows.py

echo.
echo Setup complete!
echo.
echo To run the antivirus:
echo   GUI Mode: python ai_antivirus_windows.py --gui
echo   Smart Scan: python ai_antivirus_windows.py --smart-scan
echo   Full Scan: python ai_antivirus_windows.py --full-scan
echo.
echo See WINDOWS_SETUP_GUIDE.md for detailed instructions
echo.
pause