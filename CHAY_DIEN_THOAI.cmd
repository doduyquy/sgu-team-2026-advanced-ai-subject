@echo off
setlocal
cd /d "%~dp0"
title XETUHANH - Dien thoai cam lai
if not exist ".venv\metadrive\python.exe" (
  echo Khong tim thay Python MetaDrive. Xem autonomous-driving-rl\docs\PHONE_RACING_GUIDE.md
  pause
  exit /b 1
)
".venv\metadrive\python.exe" -c "import aiohttp, aiortc, qrcode, imageio_ffmpeg; assert imageio_ffmpeg.__version__ == '0.6.0'" >nul 2>&1
if errorlevel 1 (
  echo Dang cai thu vien cho che do dien thoai...
  ".venv\metadrive\python.exe" -m pip install -r "autonomous-driving-rl\requirements-mobile.txt"
  if errorlevel 1 (
    echo Cai dat chua thanh cong. Kiem tra mang va thu lai.
    pause
    exit /b 1
  )
)
".venv\metadrive\python.exe" -B -u "autonomous-driving-rl\scripts\race_mode.py" --phone %*
if errorlevel 1 (
  echo Khong mo duoc che do dien thoai. Xem loi o tren.
  pause
  exit /b 1
)
endlocal
