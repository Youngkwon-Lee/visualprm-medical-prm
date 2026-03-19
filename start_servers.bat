@echo off
REM Start VisualPRM servers: frontend (8765) + API backend (8764)

REM Load OPENAI_API_KEY from .env if present and not already set
if not defined OPENAI_API_KEY (
    if exist .env (
        for /f "usebackq tokens=1,* delims==" %%A in (`findstr /B "OPENAI_API_KEY=" .env`) do (
            set "OPENAI_API_KEY=%%B"
        )
    )
)

REM Check for OPENAI_API_KEY environment variable
if not defined OPENAI_API_KEY (
    echo.
    echo ERROR: OPENAI_API_KEY environment variable is not set and no .env value was found!
    echo.
    echo Set it with one of:
    echo   CMD:      set OPENAI_API_KEY=sk-proj-YOUR_KEY
    echo   PowerShell: $env:OPENAI_API_KEY="sk-proj-YOUR_KEY"
    echo   OR create D:\visualprm\.env with:
    echo   OPENAI_API_KEY=sk-proj-YOUR_KEY
    echo.
    echo Then run this script again.
    pause
    exit /b 1
)

echo Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo Launching VisualPRM servers...
echo   Frontend: http://localhost:8765
echo   API Backend: http://localhost:8764
echo.
echo Press Ctrl+C to stop both servers
echo.

REM Start frontend in one terminal window (8765)
start "VisualPRM Frontend" cmd /k "python -m http.server 8765"

REM Wait a moment for frontend to start
timeout /t 2 /nobreak

REM Start API backend in another terminal window (8764)
start "VisualPRM API Backend" cmd /k "python api_backend.py"

echo.
echo Both servers are running. Close terminal windows to stop.
pause
