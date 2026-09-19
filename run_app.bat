@echo off
REM Launch the ECT scan analysis / crack detection application.
REM Double-click this file to run it.

cd /d "%~dp0"

python "tools\crack_heatmap\crack_scan_app.py"

if errorlevel 1 (
    echo.
    echo ================================================================
    echo  Could not start. Check these two things:
    echo.
    echo  1. Is Python installed?   Run:  python --version
    echo  2. Are the libraries installed?  Run:
    echo        pip install npTDMS numpy scipy matplotlib pandas anthropic
    echo.
    echo  To use the "AI assistant" tab, also set an API key:
    echo        setx ANTHROPIC_API_KEY "sk-ant-..."
    echo ================================================================
    echo.
    pause
)
