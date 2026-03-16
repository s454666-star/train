@echo off
setlocal

set "BASE_DIR=C:\Users\User\Pictures\train"
set "SERVICE1=%BASE_DIR%\start_telegram_service.bat"
set "SERVICE2=%BASE_DIR%\start_telegram_service2.bat"

call :ensure_service 8000 "%SERVICE1%"
call :ensure_service 8001 "%SERVICE2%"
exit /b 0

:ensure_service
set "TARGET_PORT=%~1"
set "TARGET_BAT=%~2"

netstat -ano | findstr /R /C:":%TARGET_PORT% .*LISTENING" >nul 2>&1
if %ERRORLEVEL%==0 (
    echo Port %TARGET_PORT% already listening. Skip %TARGET_BAT%
    goto :eof
)

if not exist "%TARGET_BAT%" (
    echo Missing startup script: %TARGET_BAT%
    goto :eof
)

echo Starting %TARGET_BAT% for port %TARGET_PORT%
start "" cmd.exe /c "%TARGET_BAT%"
goto :eof
