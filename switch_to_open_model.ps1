$src = "D:\visualprm\.env.open_model"
$dst = "D:\visualprm\.env"

if (!(Test-Path $src)) {
  Write-Error ".env.open_model not found"
  exit 1
}

Copy-Item $src $dst -Force
Write-Output "Switched .env to open_model provider"
