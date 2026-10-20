@echo off
REM ECT Crack Chat - opens the desktop program (its own window, no browser).
REM Same as double-clicking "ECT Crack Chat.lnk". The chat agent runs on this PC's
REM Claude Code login - no API key. Log: app\state\desktop.log

cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Chua co moi truong .venv. Cai dat mot lan:
    echo     python -m venv --system-site-packages .venv
    echo     .venv\Scripts\python -m pip install -r requirements.txt
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m app.desktop
