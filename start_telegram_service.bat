@echo on
chcp 65001

echo ===============================
echo START TELEGRAM FASTAPI SERVICE
echo ===============================
echo.

cd /d C:\Users\User\Pictures\train
echo Current dir:
cd
echo.

echo Using python:
C:\Users\User\Pictures\train\venv\Scripts\python.exe --version
echo.

echo Starting uvicorn...
echo.

C:\Users\User\Pictures\train\venv\Scripts\python.exe -m uvicorn telegram_service:app --host 0.0.0.0 --port 8000

echo.
echo ===============================
echo PROCESS EXITED
echo ===============================
pause
