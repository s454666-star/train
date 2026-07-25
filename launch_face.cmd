@echo off
setlocal
cd /d "%~dp0"

set "FACE_PYTHON=%~dp0.venv\Scripts\python.exe"
set "FACE_SCRIPT=%~dp0face.py"
set "TUNNEL_SCRIPT=%USERPROFILE%\.codex\skills\aws-sky\scripts\ensure_db_tunnel.ps1"
set "BLOG_ARTISAN=%USERPROFILE%\Documents\project\blog\artisan"
set "VIDEO_INDEX_ROOT=E:\video"

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

if exist "%BLOG_ARTISAN%" (
    php "%BLOG_ARTISAN%" video:repair-physical-indexes --video-root="%VIDEO_INDEX_ROOT%" --apply
    if errorlevel 1 (
        echo WARNING: Physical video DB index repair failed.
        echo face.py will open, but check the repair output above.
    )
) else (
    echo WARNING: Blog artisan was not found. Skipping physical video DB index repair:
    echo %BLOG_ARTISAN%
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
