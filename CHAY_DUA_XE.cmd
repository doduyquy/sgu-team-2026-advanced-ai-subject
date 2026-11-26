@echo off
setlocal
cd /d "%~dp0"
title XETUHANH - Race Mode
if not exist ".venv\metadrive\python.exe" (
  echo Khong tim thay Python MetaDrive tai .venv\metadrive\python.exe
  echo Xem huong dan: autonomous-driving-rl\docs\RACE_MODE_GUIDE.md
  pause
  exit /b 1
)
echo Dang mo Race Mode muot. Lan dau can nap 3 ban do va duong dua Viet Nam, vui long cho...
".venv\metadrive\python.exe" -B -u "autonomous-driving-rl\scripts\race_mode.py" %*
if errorlevel 1 (
  echo.
  echo Chuong trinh gap loi. Giu cua so nay de xem thong bao ben tren.
  pause
  exit /b 1
)
endlocal
