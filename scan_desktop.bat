@echo off
echo ========================================
echo Windows AI Antivirus - Desktop Scan
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

echo Checking virtual environment...
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created!
) else (
    echo Virtual environment found!
)

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

echo Checking dependencies...
pip show numpy >nul 2>&1
if errorlevel 1 (
    echo Installing Windows dependencies...
    pip install --upgrade pip
    pip install -r requirements_windows.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
    echo Dependencies installed successfully!
) else (
    echo Dependencies already installed!
)

echo.
echo ========================================
echo Scanning Desktop...
echo ========================================

python ai_antivirus_windows_optimized.py scan "C:\Users\ACER\Desktop" quick
if errorlevel 1 (
    echo WARNING: Desktop scan completed with errors
) else (
    echo Desktop scan completed successfully!
)

echo.
echo ========================================
echo Desktop scan finished!
echo ========================================
pause