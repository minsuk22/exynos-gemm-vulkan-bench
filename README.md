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

**A storage order.** A is column-major by default — the host uploads it
transposed, so element (m,k) sits at `A[k*M + m]` (B and C stay row-major).
This makes the m direction contiguous, which is the direction the inner loop
walks:

| | row-major A | column-major A |
|---|---|---|
| `regA[0..TM-1]` in the non-shared path | stride K apart — TM separate cache lines | one contiguous run |
| LDS staging store | transposes, so a `+4` pad is needed to break the bank conflict | already consecutive, no pad |
| LDS at 8×8 `shared_ab` | 16640 B | 16384 B |

`--a-layout row` reverts to the original order and `--a-layout both` runs the
two side by side. Because the k accumulation order is unchanged, the two
layouts must produce bit-identical results — `--mode check` verifies that,
which is also the check that the transpose itself is correct.

**Tiling:** workgroup is fixed at 16×16 invocations with a K-tile depth of 16.
The micro-tile each invocation computes is a build-time parameter: `TM`×`TN`
for `TM,TN ∈ {4,6,8}`, giving a block tile of `16·TM` × `16·TN`. All nine pairs
are built for all four staging variants and both A layouts — 72 kernels
embedded — so `--sweep` compares tiling and staging independently.

Larger micro-tiles raise arithmetic intensity: per k-step a kernel issues
`TM+TN` operand loads against `TM·TN` FMAs, so 4×4 does 2 FMA per load while
8×8 does 4. The cost is register pressure (`acc[8][8]` is 64 accumulators) and
lower occupancy, which is the tradeoff the sweep measures.

**Partial tiles.** A block tile need not divide the matrix — `TM=6` gives a
block of 96, and 96 never divides a power-of-two size. The dispatch grid is
rounded up and only the store to `C` is guarded; the loads are allowed to run
into buffer slack that is allocated and zeroed for the purpose. The guard sits
outside the k loop (one predicate per output element against `K·TM·TN` FMAs)
and is identical in all four variants, so it does not distort the comparison.
FLOP counts stay at the logical `2·M·N·K`, which charges a non-dividing tile
for the work it wastes on the padded edge rather than crediting it.

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

Requires the Android NDK and CMake + Ninja. `glslc` comes from the NDK
(`shader-tools/<host>/`), so there is no separate shader compiler to install.
Defaults point at this machine's install; override with parameters if yours
differ.

```powershell
.\build.ps1
# .\build.ps1 -Clean -NdkDir "C:\path\to\ndk"
```

### `-O` on the shader is mandatory

`glslc`/`glslangValidator` without `-O` emit a fully rolled loop nest and leave
`acc[TM][TN]`, `regA[]` and `regB[]` as function-local arrays indexed by loop
variables. Drivers park those in scratch memory, so every FMA becomes a memory
round trip — measured at **0.0008 TFLOPS**, roughly a 1000x loss. `[[unroll]]`
in the GLSL is only a hint and does not prevent this.

With `-O` the loops unroll, the arrays are promoted to SSA values, and the
inner loop becomes 256 flat FMAs with no function-local storage at all:

| | no `-O` | `-O` |
|---|---|---|
| `OpFma` | 1 | 256 |
| `TypePointer Function` | 5 | 0 |
| `OpLoopMerge` | 12 | 3 |

If you change the build, verify with
`spirv-dis foo.spv | grep -c Fma` — anything other than 256 for the default
tiling means the unroll did not happen.

### Checking which binary you are running

The binary name carries its version, and the report header repeats it along
with a scan of the embedded SPIR-V:

```
 version v0.1.2   built Aug 11 2026 13:24:07
  Embedded SPIR-V   : none=19968B/256FMA, shared_a=19196B/256FMA, ...
```

`256FMA` per kernel means the shader is optimized. Anything else (in
particular `1FMA`) prints a loud warning and means the numbers are a build
artifact, not a GPU result. `--version` prints the same summary and exits.

Output: `out\gemm_vk_bench` — a single self-contained arm64-v8a PIE. The SPIR-V
for all four kernels is embedded in the binary (no shader files to push), and
the C++ runtime is statically linked, so it only needs the platform's
`libvulkan.so` / `libc` / `libm` / `libdl`.

## Run

```powershell
.\run_on_device.ps1 -Mode perf      # timing only  (default), tile 4x4
.\run_on_device.ps1 -Mode check     # timing + result verification
.\run_on_device.ps1 -Sweep          # all 9 tiles x 4 variants, ranked
```

The sweep runs 36 kernels per size. At roughly 20 ms (2048) and 150 ms (4096)
per iteration with the default 5+2 runs, expect on the order of a minute.

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
--tile <TMxTN>        micro-tile per invocation, default 4x4. Repeatable.
--a-layout <l>        storage order of A: col (default), row, or both
--sweep               run every built (TM,TN) pair and rank the results
--samples <n>         check mode: elements verified against CPU (default 4096)
--full-check          check mode: verify the whole matrix on CPU (slow)
--device <idx>        physical device index
--list-devices        list Vulkan devices and exit
--validation          enable VK_LAYER_KHRONOS_validation
--seed <n>            RNG seed for A and B (default 12345)
--csv <path>          append per-run results to a CSV
```

Sizes must be multiples of 16 (the K tile). M and N need not divide the block
tile — see "Partial tiles" above.

## Output

Per run you get GPU time, wall time and TFLOPS; then a per-size summary with
best / median / mean, TFLOPS at best and mean, and speedup relative to the
first kernel in the list.

```
[shared_ab 8x8]  A:shared  B:shared  block 128x128  grid 16x16
    run 1/5   gpu   XXX.XXX ms   wall   XXX.XXX ms    X.XXXX TFLOPS
    ...

 rank kernel      tile block     A / B              best_ms    med_ms TFLOPS_bst   TF_wall  vs best
    1 shared_ab    8x8 128x128   shared / shared    ...
    2 ...

 BEST at 2048: shared_ab  tile 8x8 (block 128x128, A:shared B:shared)  ... TFLOPS
```

Rows are ranked by best time, so the winner is the first line and the `BEST`
line names it explicitly. Ranking is per size — the best tile at 2048 need not
be the best at 4096.

FLOP count is the standard `2*M*N*K`: 17.180 GFLOP at 2048, 137.439 GFLOP at
4096. Note that GFLOP divided by milliseconds is already TFLOP/s
(`1e9 FLOP / 1e-3 s = 1e12 FLOP/s`) — the conversion lives in one `constexpr`
guarded by a `static_assert`, because an extra `/1e3` open-coded at five call
sites once reported every result 1000x low. Timing comes from GPU timestamp queries (`VK_QUERY_TYPE_TIMESTAMP`,
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
