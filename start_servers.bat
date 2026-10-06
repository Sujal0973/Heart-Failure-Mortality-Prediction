@echo off
rem ============================================================
rem Heart Failure Mortality Prediction - Startup Script
rem ============================================================

rem Dynamically resolve WORKDIR to the script directory
set "WORKDIR=%~dp0"
if "%WORKDIR:~-1%"=="\" set "WORKDIR=%WORKDIR:~0,-1%"
set "VENV=%WORKDIR%\venv"
set "STATIC_DIR=%WORKDIR%\static"

cd /d "%WORKDIR%"

rem Determine Python and environment
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
    echo Please create a virtual environment:
    echo   python -m venv "%VENV%"
    echo   "%VENV%\Scripts\activate"
    echo   pip install -r requirements.txt
    pause
    exit /b 1
  )
  set "ACTIVATE_CMD=echo [INFO] Running in system Python"
  set "PYTHON_EXE=python"
)

rem Start uvicorn in a NEW window
start "UVICORN_SERVER" cmd /k "cd /d "%WORKDIR%" && %ACTIVATE_CMD% && uvicorn api:app --reload --port 8000"

rem Start static file server in a NEW window
start "STATIC_SERVER" cmd /k "cd /d "%STATIC_DIR%" && %PYTHON_EXE% -m http.server 8080"

rem Wait 2 seconds for servers to initialize
timeout /t 2 /nobreak >nul

rem Open browser to intro page
start "" "http://127.0.0.1:8080/intro.html"

echo ============================================================
echo Servers successfully launched:
echo  - Backend API:    http://127.0.0.1:8000 (Swagger: /docs)
echo  - Web Frontend:   http://127.0.0.1:8080/intro.html
echo ============================================================
echo To stop both servers, run stop_servers.bat
exit /b 0
