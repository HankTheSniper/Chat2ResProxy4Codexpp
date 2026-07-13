@echo off
cd /d "%~dp0"

for /f "tokens=1,2 delims==" %%a in (.env) do (
    set "%%a=%%b"
)

if "%PROXY_PORT%"=="" set "PROXY_PORT=8123"

set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY=python"

echo Killing any old proxy on port %PROXY_PORT% ...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PROXY_PORT%" ^| findstr LISTENING') do (
    taskkill /PID %%p /F >nul 2>&1
)

echo.
echo Upstream: %UPSTREAM_BASE_URL%
echo Listen  : http://127.0.0.1:%PROXY_PORT%
echo Health  : http://127.0.0.1:%PROXY_PORT%/health
echo.

"%PY%" -m uvicorn proxy:app --host 127.0.0.1 --port %PROXY_PORT%

echo.
echo ============================================================
echo  Proxy stopped or failed to start. Error above (if any).
echo ============================================================
pause
