# Push gemm_vk_bench to a connected Android device and run it.
#
#   .\run_on_device.ps1                     # perf mode, 576x160x960, 5 iters
#   .\run_on_device.ps1 -Mode check         # verify results as well
#   .\run_on_device.ps1 -ExtraArgs "--sizes 4096 --iters 50"
#
# Results are printed live and also saved to results\<timestamp>.{log,csv}.

param(
    [ValidateSet("perf", "check")]
    [string]$Mode      = "perf",
    # MxKxN, i.e. A MxK times B KxN. A bare number is a square size.
    [string]$Sizes     = "576x160x960",
    [int]$Iters        = 5,
    [int]$Warmup       = 2,
    [switch]$Sweep,        # run every built (TM,TN) tile and rank them
    [string]$ExtraArgs = "",
    [string]$Serial    = "",
    [string]$AdbExe    = "D:\Android\sdk\platform-tools\adb.exe",
    [string]$RemoteDir = "/data/local/tmp/gemm_bench"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

if (-not (Test-Path $AdbExe)) { throw "adb not found: $AdbExe" }

# Pick the newest versioned build from out\ and push it under its own name, so
# the device never holds an ambiguous "gemm_vk_bench" of unknown vintage.
$bin = Get-ChildItem (Join-Path $root "out") -Filter "gemm_vk_bench-v*" -File -ErrorAction SilentlyContinue |
       Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $bin) { throw "no binary in $root\out  (run .\build.ps1 first)" }
$remoteBin = $bin.Name
Write-Host "==> binary: $($bin.Name)" -ForegroundColor Cyan
Write-Host "    sha256  $((Get-FileHash $bin.FullName -Algorithm SHA256).Hash.ToLower())"

$adbArgs = @()
if ($Serial) { $adbArgs += @("-s", $Serial) }

$devices = & $AdbExe @adbArgs devices | Select-Object -Skip 1 | Where-Object { $_ -match "\sdevice$" }
if (-not $devices) { throw "no Android device in 'device' state. Check 'adb devices' / USB debugging." }
Write-Host "==> device: $($devices[0])" -ForegroundColor Cyan

& $AdbExe @adbArgs shell mkdir -p $RemoteDir | Out-Null
& $AdbExe @adbArgs push $bin.FullName "$RemoteDir/$remoteBin" | Out-Null
& $AdbExe @adbArgs shell chmod 755 "$RemoteDir/$remoteBin" | Out-Null

$stamp     = Get-Date -Format "yyyyMMdd-HHmmss"
$resultDir = Join-Path $root "results"
New-Item -ItemType Directory -Force $resultDir | Out-Null
$logPath   = Join-Path $resultDir "$stamp-$Mode.log"
$csvLocal  = Join-Path $resultDir "$stamp-$Mode.csv"
$csvRemote = "$RemoteDir/results.csv"

$sweepArg = if ($Sweep) { "--sweep" } else { "" }
$cmd = "cd $RemoteDir && ./$remoteBin --mode $Mode --sizes $Sizes --iters $Iters --warmup $Warmup $sweepArg --csv $csvRemote $ExtraArgs"
Write-Host "==> $cmd" -ForegroundColor Cyan
Write-Host ""

& $AdbExe @adbArgs shell $cmd 2>&1 | Tee-Object -FilePath $logPath
$rc = $LASTEXITCODE

& $AdbExe @adbArgs pull $csvRemote $csvLocal 2>&1 | Out-Null

Write-Host ""
Write-Host "==> log: $logPath" -ForegroundColor Green
if (Test-Path $csvLocal) { Write-Host "==> csv: $csvLocal" -ForegroundColor Green }
if ($rc -ne 0) { Write-Host "==> benchmark exited with code $rc" -ForegroundColor Yellow }
