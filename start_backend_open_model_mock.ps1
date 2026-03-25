$conn = Get-NetTCPConnection -LocalPort 8764 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if($conn){
  Stop-Process -Id $conn.OwningProcess -Force | Out-Null
}

$env:MODEL_PROVIDER='open_model'
$env:OPEN_MODEL_BASE_URL='http://127.0.0.1:8000/v1'
$env:OPEN_MODEL_API_KEY='EMPTY'
$env:OPEN_MODEL_GENERATE_MODEL='Qwen/Qwen2.5-VL-7B-Instruct'
$env:OPEN_MODEL_VERIFY_MODEL='Qwen/Qwen2.5-VL-7B-Instruct'

$py='C:\Users\YK\AppData\Local\Programs\Python\Python313\python.exe'
Start-Process -FilePath $py -ArgumentList 'api_backend.py' -WorkingDirectory 'D:\visualprm' -RedirectStandardOutput 'D:\visualprm\backend_open_model.log' -RedirectStandardError 'D:\visualprm\backend_open_model.err.log' | Out-Null
Start-Sleep -Seconds 2

try {
  (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8764/health -TimeoutSec 5).Content
} catch {
  $_.Exception.Message
}
