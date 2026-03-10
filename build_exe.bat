@echo off
REM =========================================================================
REM  IUTMS Windows Executable Builder
REM  ---------------------------------
REM  This script builds the IUTMS Desktop application into a standalone
REM  Windows .exe using PyInstaller.
REM
REM  Prerequisites:
REM    - Python 3.9+ installed and on PATH
REM    - pip install -r requirements.txt
REM    - pip install pyinstaller
REM
REM  Usage:
REM    build_exe.bat
REM
REM  Output:
REM    dist\IUTMS\IUTMS.exe
REM =========================================================================

echo.
echo ============================================
echo   IUTMS Desktop - Windows EXE Builder
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not on PATH.
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

REM Install dependencies
echo [1/4] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo WARNING: Some dependencies may have failed to install.
)

REM Install PyInstaller if not present
echo [2/4] Checking PyInstaller...
pip install pyinstaller
if errorlevel 1 (
    echo ERROR: Failed to install PyInstaller.
    pause
    exit /b 1
)

REM Optionally build React dashboard
if exist "web\client\package.json" (
    echo [2.5/4] Building React dashboard (optional)...
    where npm >nul 2>&1
    if not errorlevel 1 (
        cd web\client
        call npm install
        call npm run build
        cd ..\..
        echo React dashboard built successfully.
    ) else (
        echo Node.js not found - skipping React dashboard build.
        echo The desktop app will work without it.
    )
)

REM Build with PyInstaller
echo [3/4] Building executable with PyInstaller...
pyinstaller iutms.spec --noconfirm
if errorlevel 1 (
    echo ERROR: PyInstaller build failed.
    pause
    exit /b 1
)

echo.
echo [4/4] Build complete!
echo.
echo ============================================
echo   Output: dist\IUTMS\IUTMS.exe
echo ============================================
echo.
echo To run the application:
echo   1. Navigate to dist\IUTMS\
echo   2. Double-click IUTMS.exe
echo.
echo Note: SUMO must be installed separately for
echo real traffic simulations. The app works in
echo demo mode without SUMO.
echo.
echo Download SUMO: https://sumo.dlr.de/docs/Downloads.php
echo.
pause
