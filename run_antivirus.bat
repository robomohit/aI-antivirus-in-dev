@echo off
echo 🛡️ Ultimate AI Antivirus v5.X
echo ================================

echo.
echo 🔧 Activating virtual environment...
call venv\Scripts\activate

echo.
echo 🚀 Starting AI Antivirus...
echo.
echo Choose scan mode:
echo 1. GUI Mode (Recommended)
echo 2. Smart Scan
echo 3. Full Scan
echo 4. Test Suite
echo 5. Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo 🖥️ Starting GUI...
    python ai_antivirus.py --gui
) else if "%choice%"=="2" (
    echo 🧠 Starting Smart Scan...
    python ai_antivirus.py --smart-scan
) else if "%choice%"=="3" (
    echo ⚠️ WARNING: Full scan will scan your entire system!
    set /p confirm="Are you sure? (y/n): "
    if /i "%confirm%"=="y" (
        echo 🔍 Starting Full Scan...
        python ai_antivirus.py --full-scan
    ) else (
        echo ❌ Scan cancelled.
    )
) else if "%choice%"=="4" (
    echo 🧪 Running Test Suite...
    python test_suite.py --lite
) else if "%choice%"=="5" (
    echo 👋 Goodbye!
    exit /b 0
) else (
    echo ❌ Invalid choice!
)

echo.
echo ✅ Scan complete! Check logs/ folder for details.
pause