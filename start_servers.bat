@echo off
rem -------------- CONFIG --------------
rem Change WORKDIR if your project folder is elsewhere
set "WORKDIR=C:\Users\sujal\OneDrive\Documents\MINIP"
set "VENV=%WORKDIR%\venv"
set "STATIC_DIR=%WORKDIR%\static"
rem --------------------------------------

cd /d "%WORKDIR%"

rem Check venv exists
if not exist "%VENV%\Scripts\activate" (
  echo Virtual environment not found at %VENV%
  echo If you haven't created it, run:
  echo   python -m venv "%VENV%"
  echo   "%VENV%\Scripts\activate"
  echo   pip install -r requirements.txt
  pause
  exit /b 1
)

rem Start uvicorn in a NEW window (keeps it running)
start "UVICORN_SERVER" cmd /k ^"%VENV%\Scripts\activate^" ^&^& uvicorn api:app --reload --port 8000

rem Start static file server in a NEW window (serves the static folder)
start "STATIC_SERVER" cmd /k cd /d "%STATIC_DIR%" ^&^& "%VENV%\Scripts\python.exe" -m http.server 8080

rem Give servers a second to boot (optional)
timeout /t 1 /nobreak >nul

rem Open browser to intro page
start "" "http://127.0.0.1:8080/intro.html"

echo Servers started.
echo To stop both servers, run stop_servers.bat
exit /b 0
