# Build script for nonequilibrium statistical mechanics project
# Run this in VS Code terminal (PowerShell)

# Set up paths
$env:PATH = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;" + $env:PATH
$env:PATH = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;" + $env:PATH
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;" + $env:PATH

# Set up VS compiler environment by importing vcvarsall variables
$vcvars = & "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -SkipAutomaticLocation 2>&1
Write-Host "Environment loaded"

# Navigate to project
Set-Location $PSScriptRoot

# Check tools
Write-Host ""
Write-Host "=== Checking tools ==="
Write-Host "cmake: $(cmake --version 2>&1 | Select-Object -First 1)"
Write-Host "nvcc:  $(nvcc --version 2>&1 | Select-Object -Last 2 | Select-Object -First 1)"

# Configure if needed
if (-not (Test-Path "build")) {
    Write-Host ""
    Write-Host "=== Configuring (first time) ==="
    cmake -B build -G Ninja
}

# Build
Write-Host ""
Write-Host "=== Building ==="
cmake --build build

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=== BUILD SUCCESSFUL ==="
    Write-Host "  Run test:       .\build\test_diagonalize.exe"
    Write-Host "  Run simulation: .\build\quench_sim.exe"
} else {
    Write-Host ""
    Write-Host "=== BUILD FAILED ==="
}
