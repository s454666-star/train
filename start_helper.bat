@echo off
setlocal
cd /d C:\Users\Star\Documents\GitHub\train
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Star\Documents\GitHub\train\start_helper.ps1"
exit /b %ERRORLEVEL%
