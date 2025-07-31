@echo off
echo ========================================
echo Windows AI Antivirus - Downloads Scan
echo ========================================

echo.
echo Scanning Downloads folder...
python ai_antivirus_windows_optimized.py scan "C:\Users\ACER\Downloads" smart

echo.
echo Downloads scan completed!
pause