@echo off
setlocal
cd /d %~dp0

REM One-click: process ALL files in input_flacs and output vocals/instrumental pairs
set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [error] Virtual environment not found at .venv. Please install deps first:
  echo   python -m venv .venv ^&^& .\.venv\Scripts\Activate.ps1 ^&^& pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

set "INPUT_DIR=input_flacs"
if not exist "%INPUT_DIR%" mkdir "%INPUT_DIR%"

REM Optional overwrite: pass "overwrite" as arg to force regeneration
set "OVERWRITE_FLAG="
if /I "%~1"=="overwrite" set "OVERWRITE_FLAG=--overwrite"

echo Running separation on all files in %INPUT_DIR% ...
"%PY%" scripts\make_karaoke_outputs.py --input-dir "%INPUT_DIR%" --out-dir out_karaoke --model htdemucs %OVERWRITE_FLAG%
if errorlevel 1 (
  echo.
  echo [error] Separation script failed with exit code %errorlevel%.
  echo.
  pause
  exit /b %errorlevel%
)

echo.
echo Done. Outputs are saved under: %CD%\out_karaoke
echo.
pause
exit /b 0
