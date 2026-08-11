# Build gemm_vk_bench for Android arm64-v8a.
#
#   .\build.ps1                 # configure + build into build/android-arm64
#   .\build.ps1 -Clean          # wipe the build dir first
#
# Override any of the tool paths with -NdkDir / -CMakeExe / -GlslangDir if your
# SDK lives somewhere else.

param(
    [string]$NdkDir     = "D:\Android\sdk\ndk\26.3.11579264",
    [string]$CMakeExe   = "D:\Android\sdk\cmake\3.22.1\bin\cmake.exe",
    [string]$NinjaExe   = "D:\Android\sdk\cmake\3.22.1\bin\ninja.exe",
    [string]$Abi        = "arm64-v8a",
    [string]$ApiLevel   = "26",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$root     = $PSScriptRoot
$buildDir = Join-Path $root "build\android-$Abi"
$outDir   = Join-Path $root "out"

foreach ($p in @($NdkDir, $CMakeExe, $NinjaExe)) {
    if (-not (Test-Path $p)) { throw "not found: $p" }
}
# glslc ships inside the NDK; CMake finds it there.
$glslc = Join-Path $NdkDir "shader-tools\windows-x86_64\glslc.exe"
if (-not (Test-Path $glslc)) { throw "glslc not found in NDK: $glslc" }
$toolchain = Join-Path $NdkDir "build\cmake\android.toolchain.cmake"
if (-not (Test-Path $toolchain)) { throw "NDK toolchain file not found: $toolchain" }

if ($Clean -and (Test-Path $buildDir)) { Remove-Item -Recurse -Force $buildDir }
New-Item -ItemType Directory -Force $buildDir | Out-Null
New-Item -ItemType Directory -Force $outDir   | Out-Null

Write-Host "==> configuring ($Abi, android-$ApiLevel)" -ForegroundColor Cyan
& $CMakeExe -S $root -B $buildDir -G Ninja `
    "-DCMAKE_MAKE_PROGRAM=$NinjaExe" `
    "-DCMAKE_TOOLCHAIN_FILE=$toolchain" `
    "-DANDROID_ABI=$Abi" `
    "-DANDROID_PLATFORM=android-$ApiLevel" `
    "-DANDROID_STL=c++_static" `
    "-DCMAKE_BUILD_TYPE=Release" `
    "-DGLSLC=$glslc"
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

Write-Host "==> building" -ForegroundColor Cyan
& $CMakeExe --build $buildDir --parallel
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

# Name carries the version (set via OUTPUT_NAME in CMakeLists.txt). Note the
# version suffix makes PowerShell see ".2" as an extension, so match on name.
$bin = Get-ChildItem $buildDir -Filter "gemm_vk_bench-v*" -File |
       Where-Object { $_.Name -notmatch '\.(pdb|ilk|map|cmake|txt)$' } |
       Sort-Object Length -Descending | Select-Object -First 1
if (-not $bin) { throw "built binary not found in $buildDir" }

$name    = "$($bin.Name)-android-$Abi"
$outFile = Join-Path $outDir $name
Copy-Item $bin.FullName $outFile -Force

Write-Host ""
Write-Host "==> OK  $outFile  ($([math]::Round($bin.Length/1KB,1)) KB)" -ForegroundColor Green
Write-Host "    sha256 $((Get-FileHash $outFile -Algorithm SHA256).Hash.ToLower())"
Write-Host "    push and run:  .\run_on_device.ps1 -Mode perf"
