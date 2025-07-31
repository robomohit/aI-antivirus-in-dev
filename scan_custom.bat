@echo off
echo ========================================
echo Windows AI Antivirus - Custom Directory Scan
echo ========================================

if "%1"=="" (
    echo ERROR: Please provide a directory path
    echo Usage: scan_custom.bat "C:\path\to\directory" [scan_mode]
    echo.
    echo Scan modes: quick, smart, full
    echo Example: scan_custom.bat "C:\Users\ACER\Desktop" quick
    pause
    exit /b 1
)

set SCAN_PATH=%1
set SCAN_MODE=%2

if "%SCAN_MODE%"=="" set SCAN_MODE=quick

echo.
echo Target Directory: %SCAN_PATH%
echo Scan Mode: %SCAN_MODE%
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
echo Scanning: %SCAN_PATH%
echo ========================================

python ai_antivirus_windows_optimized.py scan "%SCAN_PATH%" %SCAN_MODE%
if errorlevel 1 (
    echo WARNING: Scan completed with errors
) else (
    echo Scan completed successfully!
)

echo.
echo ========================================
echo Custom scan finished!
echo ========================================
pause