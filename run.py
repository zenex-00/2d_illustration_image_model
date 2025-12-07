#!/usr/bin/env python3
"""
Run script for Gemini 3 Pro Vehicle-to-Vector API

This script starts the FastAPI server with uvicorn.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Run the FastAPI application"""
    import uvicorn
    
    # Get configuration from environment or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info")
    
    print("=" * 60)
    print("Gemini 3 Pro Vehicle-to-Vector API")
    print("=" * 60)
    print(f"Starting server on http://{host}:{port}")
    print(f"Workers: {workers}")
    print(f"Reload: {reload}")
    print(f"Log level: {log_level}")
    print("=" * 60)
    print("\nAPI Documentation:")
    print(f"  - Swagger UI: http://{host}:{port}/docs")
    print(f"  - ReDoc: http://{host}:{port}/redoc")
    print("\nWeb UI:")
    print(f"  - Home: http://{host}:{port}/ui")
    print(f"  - Training: http://{host}:{port}/ui/training")
    print(f"  - Inference: http://{host}:{port}/ui/inference")
    print("=" * 60)
    print("\nPress CTRL+C to stop the server\n")
    
    # Run uvicorn
    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,  # Reload doesn't work with multiple workers
        reload=reload,
        log_level=log_level,
        access_log=True,
        timeout_keep_alive=30,  # Connection timeout
        timeout_notify=30,      # Graceful shutdown timeout
        server_header=False
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError starting server: {e}")
        sys.exit(1)




