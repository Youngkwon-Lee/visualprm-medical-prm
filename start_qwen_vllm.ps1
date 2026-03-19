$python = "C:\Users\YK\AppData\Local\Programs\Python\Python313\python.exe"

& $python -m vllm.entrypoints.openai.api_server `
  --host 127.0.0.1 `
  --port 8000 `
  --model Qwen/Qwen2.5-VL-7B-Instruct `
  --trust-remote-code `
  --max-model-len 8192 `
  --limit-mm-per-prompt image=1
