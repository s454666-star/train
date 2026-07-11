@echo off
setlocal
cd /d C:\Users\Star\Documents\GitHub\train
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Star\Documents\GitHub\train\start_telegram_service2.ps1"
exit /b %ERRORLEVEL%
