Option Explicit

Dim shell
Dim command

command = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File ""C:\Users\Star\Documents\GitHub\train\start_telegram_service4.ps1"""

Set shell = CreateObject("WScript.Shell")
shell.Run command, 0, False
