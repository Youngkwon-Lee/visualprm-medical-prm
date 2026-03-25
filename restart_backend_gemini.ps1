$conn = Get-NetTCPConnection -LocalPort 8764 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if($conn){
  Stop-Process -Id $conn.OwningProcess -Force | Out-Null
}

$py='C:\Users\YK\AppData\Local\Programs\Python\Python313\python.exe'
Start-Process -FilePath $py -ArgumentList 'api_backend.py' -WorkingDirectory 'D:\visualprm' -RedirectStandardOutput 'D:\visualprm\backend_server.log' -RedirectStandardError 'D:\visualprm\backend_server.err.log' | Out-Null
Start-Sleep -Seconds 3

try {
  (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8764/health -TimeoutSec 5).Content
} catch {
  $_.Exception.Message
}
