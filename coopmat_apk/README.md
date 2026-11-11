# CoopMat Probe — VK_KHR_cooperative_matrix support + 2048³ GEMM per data type

An Android APK that, with one tap:

1. lists every `VkCooperativeMatrixPropertiesKHR` the GPU advertises, grouped by
   data type (A × B → C / Result, scope, saturating) with all the M×N×K shapes
   each type supports;
2. reports the features that gate each type (`shaderFloat16`, `shaderInt8`,
   `shaderBFloat16CooperativeMatrix`, `shaderFloat8CooperativeMatrix`, …);
3. runs a **2048×2048×2048 GEMM** (`D = A·B`) for every subgroup-scope
   configuration with a matching kernel. Each configuration is run once per
   memory path:

   | path | what it does |
   |---|---|
   | **no LDS** | each subgroup `coopMatLoad`s its A/B fragments straight from the global buffers |
   | **LDS x1** | single shared-memory buffer: barrier → copy slab → barrier → MMAs |
   | **LDS x2** (double) | the next slab is loaded into registers before the current slab's MMAs and written to the other buffer after them. One barrier per slab. |
   | **LDS x3** (triple) | slab t+1, already in registers, is written to LDS at the top of the iteration, then slab t+2 is fetched, then slab t is computed. The global load gets a whole compute phase to arrive, and each LDS write lands a full iteration before it is read. One barrier per slab. |

   For each path it shows TFLOPS (float) / TOPS (int), median and best GPU
   time, and a correctness check. The fastest path is marked ★.

Results are shown on screen, written to logcat (tag `COOPMAT`), and saved to
`/sdcard/Android/data/com.minsuk.coopmatprobe/files/coopmat_report.{txt,json}`.

## How each number is produced

- **Kernel**: `shaders/coopmat_gemm.comp`. It uses the same block → warp →
  fragment tiling as `VK_Test/shaders/gemm_v07.comp` (K is processed in BK
  slabs). `-DDIRECT=1` builds the no-LDS path. Otherwise the slabs are staged
  through shared memory, and the number of buffers (1/2/3) is spec constant 12,
  so a single SPIR-V covers all three LDS paths. Each extra buffer multiplies
  the LDS footprint, so the tuner drops any tiling that no longer fits
  `maxComputeSharedMemorySize`. A and B are copied as raw `uvec4`/`uint` and
  reinterpreted by `coopMatLoad`. This means the same kernel works for int8,
  fp8, bf16, fp16 and fp32 inputs, and none of them need 8-/16-bit storage
  features. The fragment shape (M, N, K) is a specialization constant, so the
  kernel runs at exactly the shape the driver advertised.
- **Tuning**: before timing, each configuration × path tries 12 tilings (6
  warp/workgroup shapes × 2 BK depths) for 3 dispatches each. Only the fastest
  one is timed. Tap a result row to see every tiling's number for every path.
- **Timing**: 2 warm-up dispatches, then 10 timed dispatches in one submit. The
  time for each dispatch is the gap between consecutive bottom-of-pipe
  timestamps. The headline number is `2·2048³ / median`.
- **Check**: inputs are random values from {-1,0,1} (or {0,1,2} for unsigned
  types). Every format holds these exactly, and every partial sum is a small
  integer. 64 outputs (two corners plus 62 random) are compared against a CPU
  reference: int and fp32 outputs must match exactly, fp16 within 1, and bf16
  within max(2, 2%). The output buffer is cleared before each path's timed run,
  so a path cannot pass by reading the previous path's output. A path that fails
  the check is shown as FAIL, not as a speed.
- **Not benchmarked** (listed with the reason): non-subgroup scopes, C ≠ Result
  type, fp64/int64, NV packed types, and any type whose feature bit is off.

## Kernels shipped

`build_shaders.sh` compiles one SPIR-V per (A, B, C) combo, with `_sat` variants
for integer types, two staging widths (`_v4` = 16 B, `_v1` = 4 B, chosen at run
time from fragment alignment) and an LDS / `_direct` pair, for 108 files in total:

fp16×fp16→fp16/fp32, fp32×fp32→fp32, bf16×bf16→bf16/fp32,
e4m3/e5m2 (all four pairings)→fp16/fp32, s8×s8→s32/s16, u8×u8→u32/s32/u16,
s8×u8→s32, u8×s8→s32.

If a device advertises a combo that isn't on this list, the app shows it as
"no prebuilt kernel". To cover it, add a line to `COMBOS` and rebuild.

## Build

Needs JDK 17, Android SDK 34 + build-tools 34, CMake 3.22.1, NDK 28.2 and
glslang ≥ 15. These are the paths on this machine:

```sh
bash build_shaders.sh                                    # -> app/src/main/assets/shaders/*.spv
JAVA_HOME="D:/Android/jdk/jdk-17.0.13+11" ./gradlew --offline assembleRelease
# -> app/build/outputs/apk/release/app-release.apk  (debug-key signed)
```

## Install / run

```sh
adb install -r app/build/outputs/apk/release/app-release.apk
adb shell am start -n com.minsuk.coopmatprobe/.MainActivity
adb logcat -s COOPMAT                                     # live progress
adb pull /sdcard/Android/data/com.minsuk.coopmatprobe/files/coopmat_report.txt
```

The phone throttles under sustained load, so later configurations in a long list
can come out a little lower. Within one configuration the paths always run in
the same order (no LDS, then x1, x2, x3), so the later paths are the ones most
exposed to throttling. To compare fairly, run again and look at the "best"
column next to the median.

## Devices tested

| Device | GPU / driver | Result |
|---|---|---|
| Galaxy S24+ (SM-S926N, Exynos 2400), Android 16 | Xclipse 940, Vulkan 1.3.279, driver 24.0.534 | `VK_KHR_cooperative_matrix` **not exposed** (0 configurations; no bf16/fp8 extensions either). The app shows the device and feature cards only. |
