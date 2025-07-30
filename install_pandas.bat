@echo off
echo Installing pandas and dependencies...
call venv\Scripts\activate.bat
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install scikit-learn>=1.3.0
python -c "import pandas; print(\"PANDAS INSTALLED!\")"
pause
