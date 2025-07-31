@echo off
echo ========================================
echo Windows AI Antivirus - Desktop Scan
echo ========================================

echo.
echo Scanning Desktop...
python ai_antivirus_windows_optimized.py scan "C:\Users\ACER\Desktop" quick

echo.
echo Desktop scan completed!
pause