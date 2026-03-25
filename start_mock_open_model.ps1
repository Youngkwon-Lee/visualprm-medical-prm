$py='C:\Users\YK\AppData\Local\Programs\Python\Python313\python.exe'
Start-Process -FilePath $py -ArgumentList 'mock_openai_compatible_server.py' -WorkingDirectory 'D:\visualprm' -RedirectStandardOutput 'D:\visualprm\mock_open_model.log' -RedirectStandardError 'D:\visualprm\mock_open_model.err.log' | Out-Null
Start-Sleep -Seconds 2
try {
  (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/health -TimeoutSec 5).Content
} catch {
  $_.Exception.Message
}
