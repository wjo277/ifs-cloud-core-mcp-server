@echo off
REM GitHub Release Upload Script for Windows
REM This script uploads the quantized models to GitHub releases

echo.
echo ==========================================
echo  GitHub Release Upload for Quantized Models
echo ==========================================
echo.

REM Check if we're in the right directory
if not exist "scripts\upload_to_github.py" (
    echo ❌ Error: Please run this script from the project root directory
    echo    Expected to find: scripts\upload_to_github.py
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    echo 🔧 Activating virtual environment...
    call .venv\Scripts\activate.bat
)

REM Run the Python upload script
echo 🚀 Starting upload process...
echo.
uv run python scripts\upload_to_github.py

REM Check exit code
if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ Upload failed with error code %ERRORLEVEL%
    echo.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ✅ Upload completed successfully!
echo.
pause
