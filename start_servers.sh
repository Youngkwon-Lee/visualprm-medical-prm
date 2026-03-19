#!/bin/bash
# Start VisualPRM servers: frontend (8765) + API backend (8764)

if [ -z "$OPENAI_API_KEY" ] && [ -f ".env" ]; then
    export OPENAI_API_KEY="$(grep '^OPENAI_API_KEY=' .env | head -n 1 | cut -d '=' -f 2-)"
fi

# Check for OPENAI_API_KEY environment variable
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "ERROR: OPENAI_API_KEY environment variable is not set and no .env value was found!"
    echo ""
    echo "Set it with:"
    echo "  export OPENAI_API_KEY=sk-proj-YOUR_KEY"
    echo "  or create D:/visualprm/.env with OPENAI_API_KEY=sk-proj-YOUR_KEY"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "Installing dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "🚀 Starting VisualPRM servers..."
echo "   Frontend: http://localhost:8765"
echo "   API Backend: http://localhost:8764"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Start frontend in background (8765)
echo "Starting frontend server..."
python -m http.server 8765 2>/dev/null &
FRONTEND_PID=$!

# Give frontend a moment to start
sleep 1

# Start API backend (8764)
echo "Starting API backend..."
python api_backend.py &
BACKEND_PID=$!

# Wait for both
wait $FRONTEND_PID $BACKEND_PID
