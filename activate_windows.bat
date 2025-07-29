@echo off
echo Activating Ultimate AI Antivirus Environment...
call venv\Scripts\activate.bat

echo.
echo Environment activated! You can now run:
echo   python ai_antivirus_windows.py --gui
echo   python ai_antivirus_windows.py --smart-scan
echo   python ai_antivirus_windows.py --full-scan
echo.
echo Or run a specific command:
if "%1"=="" (
    echo No command specified. Starting GUI...
    python ai_antivirus_windows.py --gui
) else (
    echo Running: python ai_antivirus_windows.py %*
    python ai_antivirus_windows.py %*
)