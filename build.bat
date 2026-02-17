@echo off
echo ============================================
echo  Setting up build environment...
echo ============================================

:: Set up Visual Studio compiler environment
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1

:: Add cmake and ninja to PATH (bundled with VS)
set PATH=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%PATH%
set PATH=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%

:: Navigate to project
cd /d "%~dp0"

:: Configure if build folder doesn't exist
if not exist build (
    echo ============================================
    echo  Configuring project (first time)...
    echo ============================================
    cmake -B build -G Ninja
)

:: Build
echo ============================================
echo  Building...
echo ============================================
cmake --build build

if %errorlevel% equ 0 (
    echo ============================================
    echo  Build successful!
    echo ============================================
    echo.
    echo  Run tests with:    build\test_diagonalize.exe
    echo  Run simulation:    build\quench_sim.exe
) else (
    echo ============================================
    echo  Build FAILED - check errors above
    echo ============================================
)

pause
