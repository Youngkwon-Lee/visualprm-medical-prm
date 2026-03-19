$src = "D:\visualprm\.env.commercial.example"
$dst = "D:\visualprm\.env"

if (!(Test-Path $src)) {
  Write-Error ".env.commercial.example not found"
  exit 1
}

Copy-Item $src $dst -Force
Write-Output "Switched .env to commercial provider template"
Write-Output "Fill OPENAI_API_KEY in .env before starting the backend."
