@echo off
REM Run script for Gemini 3 Pro Vehicle-to-Vector API (Windows)

REM Set default values
if "%HOST%"=="" set HOST=0.0.0.0
if "%PORT%"=="" set PORT=8000
if "%WORKERS%"=="" set WORKERS=1
if "%RELOAD%"=="" set RELOAD=false
if "%LOG_LEVEL%"=="" set LOG_LEVEL=info

echo ============================================================
echo Gemini 3 Pro Vehicle-to-Vector API
echo ============================================================
echo Starting server on http://%HOST%:%PORT%
echo Workers: %WORKERS%
echo Reload: %RELOAD%
echo Log level: %LOG_LEVEL%
echo ============================================================
echo.
echo API Documentation:
echo   - Swagger UI: http://%HOST%:%PORT%/docs
echo   - ReDoc: http://%HOST%:%PORT%/redoc
echo.
echo Web UI:
echo   - Home: http://%HOST%:%PORT%/ui
echo   - Training: http://%HOST%:%PORT%/ui/training
echo   - Inference: http://%HOST%:%PORT%/ui/inference
echo ============================================================
echo.
echo Press CTRL+C to stop the server
echo.

REM Run uvicorn
if "%RELOAD%"=="true" (
    uvicorn src.api.server:app --host %HOST% --port %PORT% --reload --log-level %LOG_LEVEL%
) else (
    uvicorn src.api.server:app --host %HOST% --port %PORT% --workers %WORKERS% --log-level %LOG_LEVEL%
)




