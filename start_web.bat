@echo off
REM Start Manga Generation Web Interface
REM This script sets up environment and starts the Flask server

echo.
echo ============================================================
echo   🎨 Manga Generation Web Interface Launcher
echo ============================================================
echo.

REM Check if venv_gpu exists
if not exist "venv_gpu" (
    echo ERROR: venv_gpu directory not found!
    echo Please run this from the project root directory.
    pause
    exit /b 1
)

REM Activate venv
echo Activating Python environment (GPU-enabled)...
call venv_gpu\Scripts\Activate.bat

REM Check for Gemini API key
if "%GEMINI_API_KEY%"=="" (
    echo.
    echo WARNING: GEMINI_API_KEY not set!
    echo You can set it with:
    echo   set GEMINI_API_KEY=your-key-here
    echo.
)

REM Install web dependencies if needed
echo Checking Flask dependencies...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo Installing Flask and dependencies...
    pip install -r requirements.txt
)

REM Set encoding
set PYTHONIOENCODING=utf-8

REM Start server
echo.
echo ============================================================
echo Starting web server...
echo Open your browser and go to: http://localhost:5000
echo Press Ctrl+C to stop
echo ============================================================
echo.

python app.py

pause
