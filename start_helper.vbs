Option Explicit

Dim shell
Dim command

command = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File ""C:\Users\User\Pictures\train\start_helper.ps1"""

Set shell = CreateObject("WScript.Shell")
shell.Run command, 0, False
