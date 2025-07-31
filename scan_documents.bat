@echo off
echo ========================================
echo Windows AI Antivirus - Documents Scan
echo ========================================

echo.
echo Scanning Documents folder...
python ai_antivirus_windows_optimized.py scan "C:\Users\ACER\Documents" full

echo.
echo Documents scan completed!
pause