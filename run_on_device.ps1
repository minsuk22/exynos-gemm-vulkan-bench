# Push gemm_vk_bench to a connected Android device and run it.
#
#   .\run_on_device.ps1                     # perf mode, 2048 + 4096, 5 iters
#   .\run_on_device.ps1 -Mode check         # verify results as well
#   .\run_on_device.ps1 -ExtraArgs "--sizes 4096 --iters 10"
#
# Results are printed live and also saved to results\<timestamp>.{log,csv}.

param(
    [ValidateSet("perf", "check")]
    [string]$Mode      = "perf",
    [string]$Sizes     = "2048,4096",
    [int]$Iters        = 5,
    [int]$Warmup       = 2,
    [string]$ExtraArgs = "",
    [string]$Serial    = "",
    [string]$AdbExe    = "D:\Android\sdk\platform-tools\adb.exe",
    [string]$RemoteDir = "/data/local/tmp/gemm_bench"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$bin  = Join-Path $root "out\gemm_vk_bench"

if (-not (Test-Path $AdbExe)) { throw "adb not found: $AdbExe" }
if (-not (Test-Path $bin))    { throw "binary not found: $bin  (run .\build.ps1 first)" }

$adbArgs = @()
if ($Serial) { $adbArgs += @("-s", $Serial) }

$devices = & $AdbExe @adbArgs devices | Select-Object -Skip 1 | Where-Object { $_ -match "\sdevice$" }
if (-not $devices) { throw "no Android device in 'device' state. Check 'adb devices' / USB debugging." }
Write-Host "==> device: $($devices[0])" -ForegroundColor Cyan

& $AdbExe @adbArgs shell mkdir -p $RemoteDir | Out-Null
& $AdbExe @adbArgs push $bin "$RemoteDir/gemm_vk_bench" | Out-Null
& $AdbExe @adbArgs shell chmod 755 "$RemoteDir/gemm_vk_bench" | Out-Null

$stamp     = Get-Date -Format "yyyyMMdd-HHmmss"
$resultDir = Join-Path $root "results"
New-Item -ItemType Directory -Force $resultDir | Out-Null
$logPath   = Join-Path $resultDir "$stamp-$Mode.log"
$csvLocal  = Join-Path $resultDir "$stamp-$Mode.csv"
$csvRemote = "$RemoteDir/results.csv"

$cmd = "cd $RemoteDir && ./gemm_vk_bench --mode $Mode --sizes $Sizes --iters $Iters --warmup $Warmup --csv $csvRemote $ExtraArgs"
Write-Host "==> $cmd" -ForegroundColor Cyan
Write-Host ""

& $AdbExe @adbArgs shell $cmd 2>&1 | Tee-Object -FilePath $logPath
$rc = $LASTEXITCODE

& $AdbExe @adbArgs pull $csvRemote $csvLocal 2>&1 | Out-Null

Write-Host ""
Write-Host "==> log: $logPath" -ForegroundColor Green
if (Test-Path $csvLocal) { Write-Host "==> csv: $csvLocal" -ForegroundColor Green }
if ($rc -ne 0) { Write-Host "==> benchmark exited with code $rc" -ForegroundColor Yellow }
