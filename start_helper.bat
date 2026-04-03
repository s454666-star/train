@echo off
setlocal
cd /d C:\Users\User\Pictures\train
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\User\Pictures\train\start_helper.ps1"
exit /b %ERRORLEVEL%
