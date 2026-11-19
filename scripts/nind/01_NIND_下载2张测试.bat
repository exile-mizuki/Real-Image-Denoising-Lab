@echo off
setlocal EnableExtensions
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "DATA=%~dp0..\..\datasets\NIND"
set "OUT=%~dp0..\..\outputs\NIND"
if defined DENOISE_DATA_DIR set "DATA=%DENOISE_DATA_DIR%"
if not exist "%DATA%" (
  echo [ERROR] Dataset directory not found: "%DATA%"
  echo Put the dataset in datasets/ or set DENOISE_DATA_DIR to your actual dataset path.
  pause
  exit /b 1
)
if not exist "%OUT%" mkdir "%OUT%"
where py >nul 2>&1
if not errorlevel 1 (
  py -3 -X utf8 "%~dp0download_nind.py" --root "%DATA%" --limit 2
) else (
  python -X utf8 "%~dp0download_nind.py" --root "%DATA%" --limit 2
)
if errorlevel 1 (echo [ERROR] See details above.) else (echo SUCCESS.)
pause
endlocal
