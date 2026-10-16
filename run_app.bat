@echo off
REM ECT Crack Chat - double-click to start the backend and open the app window.
REM The chat agent runs on this PC's Claude Code login (no API key needed).
REM Close the minimized "ECT backend" window to stop the app.

cd /d "%~dp0"
set ECT_PORT=8765

if not exist ".venv\Scripts\python.exe" (
    echo Chua co moi truong .venv. Cai dat mot lan:
    echo     python -m venv --system-site-packages .venv
    echo     .venv\Scripts\python -m pip install -r requirements.txt
    pause
    exit /b 1
)
if not exist "app\state" mkdir "app\state"

REM Already running? Then just open the window.
powershell -NoProfile -Command "try{Invoke-WebRequest -UseBasicParsing http://127.0.0.1:%ECT_PORT%/api/health -TimeoutSec 1 | Out-Null; exit 0}catch{exit 1}"
if not errorlevel 1 goto open

start "ECT backend" /min cmd /c ".venv\Scripts\python.exe -m app.backend.server > app\state\server.log 2>&1"

REM Wait up to 30 s for the backend to answer.
powershell -NoProfile -Command "for($i=0;$i -lt 60;$i++){try{Invoke-WebRequest -UseBasicParsing http://127.0.0.1:%ECT_PORT%/api/health -TimeoutSec 1 | Out-Null; exit 0}catch{Start-Sleep -Milliseconds 500}}; exit 1"
if errorlevel 1 (
    echo Backend khong khoi dong duoc. Xem app\state\server.log
    type app\state\server.log
    pause
    exit /b 1
)

:open
REM App-style window without an address bar (Edge ships with Windows 11).
start "" msedge --app=http://127.0.0.1:%ECT_PORT% --window-size=1500,900
