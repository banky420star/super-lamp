$env:AGI_HOST = "0.0.0.0"
$env:AGI_PORT = if ($env:AGI_PORT) { $env:AGI_PORT } else { "9090" }
$token = $env:AGI_TOKEN
if (-not $token) {
    Write-Host "WARNING: AGI_TOKEN not set in environment. Using placeholder for local/dev only." -ForegroundColor Yellow
    $token = "dev-placeholder-do-not-use-in-prod"
}
$env:AGI_TOKEN = $token

$env:AGI_AUTONOMY_AUTO_CANARY = if ($env:AGI_AUTONOMY_AUTO_CANARY) { $env:AGI_AUTONOMY_AUTO_CANARY } else { "true" }
$env:AGI_PNL_POLL = if ($env:AGI_PNL_POLL) { $env:AGI_PNL_POLL } else { "true" }

$env:AGI_COOLDOWN_SEC = if ($env:AGI_COOLDOWN_SEC) { $env:AGI_COOLDOWN_SEC } else { "45" }
$env:AGI_MIN_HOLD_SEC = if ($env:AGI_MIN_HOLD_SEC) { $env:AGI_MIN_HOLD_SEC } else { "120" }
$env:CANARY_LOT_MULT = if ($env:CANARY_LOT_MULT) { $env:CANARY_LOT_MULT } else { "0.25" }

$env:AGI_DZ_EURUSD = if ($env:AGI_DZ_EURUSD) { $env:AGI_DZ_EURUSD } else { "0.18" }
$env:AGI_DZ_GBPUSD = if ($env:AGI_DZ_GBPUSD) { $env:AGI_DZ_GBPUSD } else { "0.20" }
$env:AGI_DZ_XAUUSD = if ($env:AGI_DZ_XAUUSD) { $env:AGI_DZ_XAUUSD } else { "0.22" }

Write-Host "Starting Grok AGI Server on Port $($env:AGI_PORT) ..."
$py = Join-Path $PSScriptRoot '.venv312\Scripts\python.exe'
if (-not (Test-Path $py)) {
    $py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
}
if (-not (Test-Path $py)) {
    throw "No venv python found (.venv312 or .venv). Run: python -m venv .venv312 ; .venv312\Scripts\Activate.ps1 ; pip install -r requirements.txt"
}
$env:CHAIN_GAMBLER_EXECUTION_MODE = "paper"
$env:CHAIN_GAMBLER_ALLOW_LIVE = "0"
& $py -m Python.Server_AGI
