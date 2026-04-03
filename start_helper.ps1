param(
    [string]$HelperHost = '127.0.0.1',
    [int]$Port = 8787,
    [string]$IdmPath = ''
)

$ErrorActionPreference = 'Stop'

$baseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $baseDir 'venv\Scripts\python.exe'
$pythonwExe = Join-Path $baseDir 'venv\Scripts\pythonw.exe'
$helperScript = Join-Path $baseDir 'helper_server.py'
$stdoutLog = Join-Path $baseDir ("logs\helper_{0}.stdout.log" -f $Port)
$stderrLog = Join-Path $baseDir ("logs\helper_{0}.stderr.log" -f $Port)

function Test-HelperHealthy {
    param(
        [string]$TargetHost,
        [int]$TargetPort
    )

    try {
        $response = Invoke-RestMethod -Uri ("http://{0}:{1}/health" -f $TargetHost, $TargetPort) -Method Get -TimeoutSec 2
        return [bool]$response.ok
    } catch {
        return $false
    }
}

function Wait-ForHelper {
    param(
        [string]$TargetHost,
        [int]$TargetPort,
        [int]$TimeoutSeconds = 20
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-HelperHealthy -TargetHost $TargetHost -TargetPort $TargetPort) {
            return $true
        }

        Start-Sleep -Milliseconds 300
    }

    return $false
}

if (-not (Test-Path -LiteralPath $helperScript)) {
    throw "Helper script not found: $helperScript"
}

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python executable not found: $pythonExe"
}

if (Test-HelperHealthy -TargetHost $HelperHost -TargetPort $Port) {
    exit 0
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $stdoutLog) | Out-Null

$argumentList = @($helperScript, '--host', $HelperHost, '--port', [string]$Port)
if ($IdmPath) {
    $argumentList += @('--idm-path', $IdmPath)
}

if (Test-Path -LiteralPath $pythonwExe) {
    $process = Start-Process `
        -FilePath $pythonwExe `
        -ArgumentList $argumentList `
        -WorkingDirectory $baseDir `
        -WindowStyle Hidden `
        -PassThru
} else {
    $process = Start-Process `
        -FilePath $pythonExe `
        -ArgumentList $argumentList `
        -WorkingDirectory $baseDir `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -WindowStyle Hidden `
        -PassThru
}

if (-not (Wait-ForHelper -TargetHost $HelperHost -TargetPort $Port)) {
    try {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    } catch {
    }

    throw "IDM helper did not become healthy on $HelperHost`:$Port."
}
