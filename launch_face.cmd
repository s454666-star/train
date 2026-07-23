@echo off
setlocal
cd /d "%~dp0"

set "FACE_PYTHON=%~dp0.venv\Scripts\python.exe"
set "FACE_SCRIPT=%~dp0face.py"
set "TUNNEL_SCRIPT=%USERPROFILE%\.codex\skills\aws-sky\scripts\ensure_db_tunnel.ps1"

if not exist "%FACE_PYTHON%" (
    echo ERROR: Project Python was not found:
    echo %FACE_PYTHON%
    echo.
    pause
    exit /b 1
)

if not exist "%FACE_SCRIPT%" (
    echo ERROR: face.py was not found:
    echo %FACE_SCRIPT%
    echo.
    pause
    exit /b 1
)

if exist "%TUNNEL_SCRIPT%" (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%TUNNEL_SCRIPT%"
    if errorlevel 1 (
        echo WARNING: The AWS database tunnel could not be started.
        echo face.py will open, but database operations may fail.
    )
)

"%FACE_PYTHON%" "%FACE_SCRIPT%"
set "FACE_EXIT_CODE=%ERRORLEVEL%"

if not "%FACE_EXIT_CODE%"=="0" (
    echo.
    echo face.py failed with exit code %FACE_EXIT_CODE%.
    echo Check face_extractor_runtime.log for details.
    pause
)

exit /b %FACE_EXIT_CODE%
