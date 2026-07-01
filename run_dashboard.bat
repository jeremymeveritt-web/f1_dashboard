@echo off
echo ===================================================
echo Starting F1 Analytics Dashboard...
echo ===================================================

:: Force the script to run from the folder it lives in
cd /d "%~dp0"

:: Temporarily bypass execution restrictions and activate the virtual environment
cmd /k ".\venv\Scripts\activate.bat && python app.py"

pause