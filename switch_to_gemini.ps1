$src = "D:\visualprm\.env.gemini.example"
$dst = "D:\visualprm\.env"

if (!(Test-Path $src)) {
  Write-Error ".env.gemini.example not found"
  exit 1
}

Copy-Item $src $dst -Force
Write-Output "Switched .env to gemini provider template"
Write-Output "Fill GEMINI_API_KEY in .env before starting the backend."
