@echo off
rem ============================================================
rem CardioPulse AI - Heart Failure Mortality Prediction
rem Unified Fast Server Launch Script (Windows)
rem ============================================================

set "WORKDIR=%~dp0"
if "%WORKDIR:~-1%"=="\" set "WORKDIR=%WORKDIR:~0,-1%"
set "VENV=%WORKDIR%\venv"
set "STATIC_DIR=%WORKDIR%\static"

cd /d "%WORKDIR%"

rem Determine Python and virtual environment
if exist "%VENV%\Scripts\activate.bat" (
  echo [INFO] Activating virtual environment: %VENV%
  set "ACTIVATE_CMD=call "%VENV%\Scripts\activate.bat""
  set "PYTHON_EXE="%VENV%\Scripts\python.exe""
) else (
  echo [INFO] Virtual environment not found at %VENV%.
  echo [INFO] Checking for system Python...
  where python >nul 2>nul
  if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH or virtual environment!
    echo Please install Python 3.10+ or create a virtual environment:
    echo   python -m venv "%VENV%"
    echo   "%VENV%\Scripts\activate"
    echo   pip install -r requirements.txt
    pause
    exit /b 1
  )
  set "ACTIVATE_CMD=echo [INFO] Running in system Python"
  set "PYTHON_EXE=python"
)

rem 1. Start FastAPI Backend & Static Application on Port 8000
start "CARDIOPULSE_BACKEND_8000" cmd /k "cd /d "%WORKDIR%" && %ACTIVATE_CMD% && uvicorn api:app --reload --port 8000"

rem 2. Start Static File Server on Port 8080 for Standalone Web Access
start "CARDIOPULSE_FRONTEND_8080" cmd /k "cd /d "%STATIC_DIR%" && %PYTHON_EXE% -m http.server 8080"

rem Wait 2 seconds for services to bind
timeout /t 2 /nobreak >nul

rem Open default browser to the interactive dashboard
start "" "http://127.0.0.1:8000/dashboard"

echo ============================================================
echo   CardioPulse AI Servers Successfully Launched!
echo ============================================================
echo  - Unified App & API:   http://127.0.0.1:8000/dashboard
echo  - Interactive Docs:    http://127.0.0.1:8000/docs
echo  - Welcome Portal:      http://127.0.0.1:8080/intro.html
echo  - Clinical Guidelines:  http://127.0.0.1:8080/dos_donts.html
echo ============================================================
echo To stop all running servers, execute: stop_servers.bat
exit /b 0
