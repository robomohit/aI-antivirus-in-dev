@echo off
echo ========================================
echo Windows AI Antivirus Setup
echo ========================================

echo.
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found!
echo.

echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Virtual environment created!
echo.

echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo Virtual environment activated!
echo.

echo Installing Windows dependencies...
pip install --upgrade pip
pip install -r requirements_windows.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo Dependencies installed successfully!
echo.

echo Testing antivirus...
python ai_antivirus_windows_optimized.py scan . quick
if errorlevel 1 (
    echo WARNING: Antivirus test failed, but setup completed
) else (
    echo Antivirus test successful!
)
echo.
echo NOTE: Using ai_antivirus_windows_optimized.py (Windows optimized version)
echo If you see false positives, make sure you're using this file!
echo.
echo ========================================
echo SCANNING SPECIFIC DIRECTORIES
echo ========================================
echo.
echo To scan a specific directory, use:
echo python ai_antivirus_windows_optimized.py scan "C:\Users\ACER\Desktop" quick
echo.
echo Examples:
echo - Desktop: python ai_antivirus_windows_optimized.py scan "C:\Users\ACER\Desktop" quick
echo - Downloads: python ai_antivirus_windows_optimized.py scan "C:\Users\ACER\Downloads" smart
echo - Documents: python ai_antivirus_windows_optimized.py scan "C:\Users\ACER\Documents" full
echo.

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To use the antivirus:
echo 1. Activate virtual environment: venv\Scripts\activate.bat
echo 2. Quick scan: python ai_antivirus_windows_optimized.py scan . quick
echo 3. Smart scan: python ai_antivirus_windows_optimized.py scan . smart
echo 4. Full scan: python ai_antivirus_windows_optimized.py scan . full
echo 5. Monitor mode: python ai_antivirus_windows_optimized.py monitor .
echo.
pause