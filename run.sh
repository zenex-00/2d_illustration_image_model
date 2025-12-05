#!/bin/bash
# Run script for Gemini 3 Pro Vehicle-to-Vector API (Linux/Mac)

# Set default values
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
WORKERS=${WORKERS:-1}
RELOAD=${RELOAD:-false}
LOG_LEVEL=${LOG_LEVEL:-info}

echo "============================================================"
echo "Gemini 3 Pro Vehicle-to-Vector API"
echo "============================================================"
echo "Starting server on http://$HOST:$PORT"
echo "Workers: $WORKERS"
echo "Reload: $RELOAD"
echo "Log level: $LOG_LEVEL"
echo "============================================================"
echo ""
echo "API Documentation:"
echo "  - Swagger UI: http://$HOST:$PORT/docs"
echo "  - ReDoc: http://$HOST:$PORT/redoc"
echo ""
echo "Web UI:"
echo "  - Home: http://$HOST:$PORT/ui"
echo "  - Training: http://$HOST:$PORT/ui/training"
echo "  - Inference: http://$HOST:$PORT/ui/inference"
echo "============================================================"
echo ""
echo "Press CTRL+C to stop the server"
echo ""

# Run uvicorn
if [ "$RELOAD" = "true" ]; then
    uvicorn src.api.server:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --log-level "$LOG_LEVEL"
else
    uvicorn src.api.server:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL"
fi




