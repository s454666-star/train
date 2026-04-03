param(
    [Parameter(Mandatory = $true)]
    [string]$ModuleName,
    [Parameter(Mandatory = $true)]
    [int]$Port
)

$ErrorActionPreference = 'Stop'

$baseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $baseDir 'venv\Scripts\python.exe'
$stdoutLog = Join-Path $baseDir ("logs\telegram_service_{0}.stdout.log" -f $Port)
$stderrLog = Join-Path $baseDir ("logs\telegram_service_{0}.stderr.log" -f $Port)

function Test-PortListening {
    param(
        [int]$TargetPort
    )

    try {
        return [bool](Get-NetTCPConnection -LocalPort $TargetPort -State Listen -ErrorAction Stop)
    } catch {
        return $false
    }
}

function Wait-ForTcpPort {
    param(
        [int]$TargetPort,
        [int]$TimeoutSeconds = 20
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-PortListening -TargetPort $TargetPort) {
            return $true
        }

        Start-Sleep -Milliseconds 300
    }

    return $false
}

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python executable not found: $pythonExe"
}

if (-not (Test-Path -LiteralPath (Join-Path $baseDir ("{0}.py" -f $ModuleName)))) {
    throw "Telegram service wrapper not found for module: $ModuleName"
}

if (Test-PortListening -TargetPort $Port) {
    exit 0
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $stdoutLog) | Out-Null

$process = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList @('-m', 'uvicorn', "$ModuleName`:app", '--host', '0.0.0.0', '--port', [string]$Port) `
    -WorkingDirectory $baseDir `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru

if (-not (Wait-ForTcpPort -TargetPort $Port)) {
    try {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    } catch {
    }

    throw "Telegram FastAPI module $ModuleName did not open port $Port."
}
