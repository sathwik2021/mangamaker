# Start Manga Generation Web Interface
# PowerShell launch script

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   🎨 Manga Generation Web Interface Launcher" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if venv_gpu exists
if (-not (Test-Path "venv_gpu")) {
    Write-Host "ERROR: venv_gpu directory not found!" -ForegroundColor Red
    Write-Host "Please run this from the project root directory." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate venv
Write-Host "Activating Python environment (GPU-enabled)..." -ForegroundColor Green
& ".\venv_gpu\Scripts\Activate.ps1"

# Check for Gemini API key
if (-not $env:GEMINI_API_KEY) {
    Write-Host ""
    Write-Host "WARNING: GEMINI_API_KEY not set!" -ForegroundColor Yellow
    Write-Host "You can set it with:"
    Write-Host "  `$env:GEMINI_API_KEY = 'your-key-here'" -ForegroundColor Gray
    Write-Host ""
}

# Set encoding
$env:PYTHONIOENCODING = "utf-8"

# Install web dependencies if needed
Write-Host "Checking Flask dependencies..." -ForegroundColor Yellow
$flaskInstalled = pip show flask 2>$null
if (-not $flaskInstalled) {
    Write-Host "Installing Flask and dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Starting web server..." -ForegroundColor Green
Write-Host "Open your browser and go to: http://localhost:5000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Start Flask app
python app.py

Write-Host ""
Write-Host "Server stopped." -ForegroundColor Yellow
Read-Host "Press Enter to exit"
