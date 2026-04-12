param(
  [int]$NEach = 5,
  [string]$PrmEndpoint = ""
)

if ($PrmEndpoint -eq "") {
  Write-Host "Usage: ./run_eval_prm_real.ps1 -PrmEndpoint <url> [-NEach 5]"
  exit 1
}

$env:PRM_PROVIDER = "http"
$env:PRM_HTTP_ENDPOINT = $PrmEndpoint
$env:PRM_HTTP_TIMEOUT_SEC = "15"

Write-Host "PRM_PROVIDER=$env:PRM_PROVIDER"
Write-Host "PRM_HTTP_ENDPOINT=$env:PRM_HTTP_ENDPOINT"

py -3 prm_mode_compare.py
