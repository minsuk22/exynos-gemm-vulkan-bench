# gemm_vk_bench — Vulkan SGEMM benchmark for Exynos (Xclipse) GPUs

Android arm64 command-line binary that measures fp32 `C[MxN] = A[MxK] * B[KxN]`
on a Vulkan compute queue, comparing four kernel variants that differ **only**
in whether the A / B operand is staged through workgroup shared memory (LDS).

| kernel      | A operand | B operand | LDS used |
|-------------|-----------|-----------|----------|
| `none`      | global    | global    | 0 B      |
| `shared_a`  | shared    | global    | 4352 B   |
| `shared_b`  | global    | shared    | 4096 B   |
| `shared_ab` | shared    | shared    | 8448 B   |

All four come from the same source (`shaders/gemm.comp`), selected at SPIR-V
compile time with `-DUSE_SHARED_A` / `-DUSE_SHARED_B`. Tiling, register
blocking, loop order and FMA order are identical, so the delta you measure is
the operand-staging strategy and nothing else.

**Tiling:** workgroup 16×16 invocations, 64×64 output tile per workgroup, 4×4
outputs per invocation, K-tile depth 16.

Because the FMA order is identical everywhere, all four kernels should produce
**bit-identical** results — the benchmark checks that too.

---

## Download

A prebuilt arm64-v8a binary is attached to the
[latest release](https://github.com/minsuk22/exynos-gemm-vulkan-bench/releases/latest)
if you don't want to set up the NDK:

```sh
curl -L -o gemm_vk_bench \
  https://github.com/minsuk22/exynos-gemm-vulkan-bench/releases/latest/download/gemm_vk_bench-android-arm64-v8a
adb push gemm_vk_bench /data/local/tmp/
adb shell chmod 755 /data/local/tmp/gemm_vk_bench
adb shell /data/local/tmp/gemm_vk_bench --mode check --sizes 2048 --iters 2
```

## Build

Requires the Android NDK, CMake + Ninja, and `glslangValidator`. Defaults point
at this machine's install; override with parameters if yours differ.

```powershell
.\build.ps1
# .\build.ps1 -Clean -NdkDir "C:\path\to\ndk" -GlslangDir "C:\VulkanSDK\1.3.x"
```

Output: `out\gemm_vk_bench` — a single self-contained arm64-v8a PIE. The SPIR-V
for all four kernels is embedded in the binary (no shader files to push), and
the C++ runtime is statically linked, so it only needs the platform's
`libvulkan.so` / `libc` / `libm` / `libdl`.

## Run

```powershell
.\run_on_device.ps1 -Mode perf      # timing only  (default)
.\run_on_device.ps1 -Mode check     # timing + result verification
```

The script pushes the binary to `/data/local/tmp/gemm_bench`, runs it, and
saves the console log plus a CSV under `results\`.

Manual equivalent (any OS with adb):

```sh
adb push out/gemm_vk_bench /data/local/tmp/
adb shell chmod 755 /data/local/tmp/gemm_vk_bench
adb shell /data/local/tmp/gemm_vk_bench --mode perf
```

## Modes

* **`--mode perf`** (default) — pure performance. `C` is never read back and no
  CPU reference is computed, so nothing perturbs the measurement.
* **`--mode check`** — same benchmark, then `C` is copied back and verified:
  1. against a CPU reference computed in `double` for randomly sampled output
     elements (default 4096, plus all four corners), and
  2. against the first kernel's full output buffer, element for element.

  Tolerance is `|err| <= 1e-2 + 1e-4*|ref|`, comfortably above the ~1e-4
  absolute error fp32 accumulation over K=4096 actually produces, and far below
  any real indexing or race bug. `--full-check` verifies every element with a
  multithreaded CPU GEMM instead of sampling (slow — 137 GFLOP at 4096).

## Options

```
--mode <perf|check>   run mode (default perf)
--sizes <list>        comma separated square sizes (default 2048,4096)
--iters <n>           timed iterations per kernel (default 5)
--warmup <n>          untimed warmup iterations (default 2)
--kernels <list>      subset of none,shared_a,shared_b,shared_ab
--samples <n>         check mode: elements verified against CPU (default 4096)
--full-check          check mode: verify the whole matrix on CPU (slow)
--device <idx>        physical device index
--list-devices        list Vulkan devices and exit
--validation          enable VK_LAYER_KHRONOS_validation
--seed <n>            RNG seed for A and B (default 12345)
--csv <path>          append per-run results to a CSV
```

Sizes must be multiples of 64 (M/N tile) and 16 (K tile). 2048 and 4096 both
qualify.

## Output

Per run you get GPU time, wall time and TFLOPS; then a per-size summary with
best / median / mean, TFLOPS at best and mean, and speedup relative to the
first kernel in the list.

```
[shared_ab]  A:shared  B:shared
    run 1/5   gpu   XXX.XXX ms   wall   XXX.XXX ms    X.XXXX TFLOPS
    ...

 kernel     A / B             best_ms    med_ms    avg_ms TFLOPS_bst TFLOPS_avg  vs base
 none       global / global   ...
 shared_a   shared / global   ...
 shared_b   global / shared   ...
 shared_ab  shared / shared   ...
```

FLOP count is the standard `2*M*N*K`: 17.180 GFLOP at 2048, 137.439 GFLOP at
4096. Timing comes from GPU timestamp queries (`VK_QUERY_TYPE_TIMESTAMP`,
scaled by `timestampPeriod`) when the queue family supports them, falling back
to wall-clock around submit→fence otherwise; the summary states which was used.

## Notes on measurement

* Phones throttle. The 4096 case moves 192 MiB of buffers and runs the GPU flat
  out; run it twice and compare if the numbers look unstable, or raise
  `--iters` and read the median rather than the mean.
* Memory: each size allocates 3 device buffers plus one host-visible staging
  buffer, i.e. 4×64 MiB at 4096. If allocation fails on a memory-constrained
  device, run the sizes separately (`--sizes 4096`).
* `--mode check` allocates two extra host copies of A and B (128 MiB at 4096)
  for the CPU reference. Use `--mode perf` for the numbers you report.
