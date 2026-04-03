$ErrorActionPreference = 'Stop'

$baseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$servicesLauncher = Join-Path $baseDir 'start_telegram_services.vbs'
$startupDir = [Environment]::GetFolderPath('Startup')
$startupLauncher = Join-Path $startupDir 'Train Telegram Services Startup.vbs'

if (-not (Test-Path -LiteralPath $servicesLauncher)) {
    throw "Services launcher not found: $servicesLauncher"
}

$startupContent = @"
Option Explicit

Dim shell
Dim command

command = "wscript.exe ""$servicesLauncher"""

Set shell = CreateObject("WScript.Shell")
shell.Run command, 0, False
"@

[System.IO.File]::WriteAllText($startupLauncher, $startupContent, [System.Text.Encoding]::ASCII)
Write-Host "Startup launcher written: $startupLauncher"

$watchdogTaskName = 'Telegram FastAPI Services'
try {
    $null = Get-ScheduledTask -TaskName $watchdogTaskName -ErrorAction Stop
    $action = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument ('"{0}"' -f $servicesLauncher)
    Set-ScheduledTask -TaskName $watchdogTaskName -Action $action | Out-Null
    Write-Host "Watchdog task updated: $watchdogTaskName"
} catch {
    Write-Warning ("Unable to update watchdog task {0}: {1}" -f $watchdogTaskName, $_.Exception.Message)
}
