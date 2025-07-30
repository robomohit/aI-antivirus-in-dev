@echo off
echo ================================================
echo ULTIMATE AI ANTIVIRUS - WINDOWS SETUP FIX
echo ================================================

echo.
echo Step 1: Checking Python...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo.
echo Step 2: Removing old virtual environment...
if exist "venv" (
    echo Removing old venv...
    rmdir /s /q venv
)

echo.
echo Step 3: Creating fresh virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment!
    pause
    exit /b 1
)

echo.
echo Step 4: Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)

echo.
echo Step 5: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Step 6: Installing pandas first...
pip install pandas>=2.0.0
if %errorlevel% neq 0 (
    echo ERROR: Failed to install pandas!
    pause
    exit /b 1
)

echo.
echo Step 7: Installing numpy...
pip install numpy>=1.24.0

echo.
echo Step 8: Installing scikit-learn...
pip install scikit-learn>=1.3.0

echo.
echo Step 9: Installing torch...
pip install torch>=2.0.0

echo.
echo Step 10: Installing other dependencies...
pip install watchdog>=3.0.0
pip install colorama>=0.4.6
pip install rich>=13.0.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install shap>=0.41.0

echo.
echo Step 11: Creating directories...
if not exist "model" mkdir model
if not exist "logs" mkdir logs
if not exist "quarantine" mkdir quarantine
if not exist "test_files" mkdir test_files

echo.
echo Step 12: Testing pandas installation...
python -c "import pandas; print('PANDAS INSTALLED SUCCESSFULLY!')"
if %errorlevel% neq 0 (
    echo ERROR: Pandas test failed!
    pause
    exit /b 1
)

echo.
echo Step 13: Creating dataset...
python create_dataset_windows.py

echo.
echo Step 14: Training model...
python train_enhanced_model_windows.py

echo.
echo ================================================
echo SETUP COMPLETE!
echo ================================================
echo.
echo To run the antivirus:
echo   python ai_antivirus_windows.py --smart-scan
echo   python ai_antivirus_windows.py --full-scan
echo   python ai_antivirus_windows.py --gui
echo.
pause