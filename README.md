# gemm_vk_bench — Vulkan SGEMM benchmark for Exynos (Xclipse) GPUs

Android arm64 command-line binary that measures fp32 `C[MxN] = A[MxK] * B[KxN]`
on a Vulkan compute queue, comparing four kernel variants that differ **only**
in whether the A / B operand is staged through workgroup shared memory (LDS).

**[Quick start](#quick-start)** if you just want the prebuilt binary on a phone
— four commands, no NDK. What comes first is what it measures and why.

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
are built for all four staging variants, both A layouts, both double-buffer
modes and — where it differs — both B column mappings, so `--sweep` compares
tiling, staging, layout, buffering and the B mapping independently. 168 kernels
are embedded.

Larger micro-tiles raise arithmetic intensity: per k-step a kernel issues
`TM+TN` operand loads against `TM·TN` FMAs, so 4×4 does 2 FMA per load while
8×8 does 4. The cost is register pressure (`acc[8][8]` is 64 accumulators) and
lower occupancy, which is the tradeoff the sweep measures.

**LDS double buffering.** With `--double-buffer on`, the shared-memory tiles are
ping-ponged: tile `kt` is read out of buffer `cur` while tile `kt+BK` is fetched
from global into registers and written to buffer `cur^1`. Because the two
buffers never alias, **one barrier per k-iteration suffices instead of two** —
it both publishes the writes to `cur^1` and confirms every invocation has
finished reading `cur`. Verified in the SPIR-V: the single-buffered kernel has
both `OpControlBarrier`s inside the k loop, the double-buffered one has a
prologue barrier outside it and only one inside.

The prefetch is issued before the FMAs so the global load latency overlaps with
compute. What it costs is registers and LDS:

| shared_ab 8×8, A-col | single | double |
|---|---|---|
| barriers per k-iteration | 2 | 1 |
| LDS | 16384 B | 32768 B |
| prefetch registers held across compute | 0 | 16 |

Doubling LDS and adding 16 live registers on top of `acc[8][8]` cuts occupancy,
so double buffering is not automatically a win — which is why it is a
comparison axis (`off` / `on` / `both`, default `both`) rather than a
replacement. Combinations that no longer fit in the device's shared memory are
skipped with a printed reason instead of aborting the run; `shared_ab` 8×8 with
a row-major A needs 33280 B and will be dropped on a 32 KiB device.

`none` has no LDS, so it has no double-buffered build at all.

**B column mapping.** Which of the block tile's `BN` columns an invocation owns
is the other thing a `TN`-wide micro-tile gets to choose. By default invocation
`x` owns one run, `x·TN … x·TN+TN-1`. Its own loads for a given `k` are
contiguous, but the four a 16-byte transaction covers sit `TN` floats away from
invocation `x+1`'s four, so the wave's first quarter of loads is strided across
the whole tile rather than packed.

`--b-split on` cuts the run into groups of four and spreads the groups: group
`g` of invocation `x` lands at `g·(16·4) + x·4`. Consecutive invocations are
then exactly four floats apart inside a group, so each group of loads is one
unbroken run of 64 columns. Written out for a two-invocation tile:

```
off   x=0: 0 1 2 3 4 5 6 7    x=1: 8 9 A B C D E F
on    x=0: 0 1 2 3 8 9 A B    x=1: 4 5 6 7 C D E F
```

`C` is stored through the same mapping, so the matrix produced is unchanged and
still bit-identical — only which invocation owns which column moves, and with
it the address pattern of the global `B` loads (`none`, `shared_a`) or of the
LDS reads (`shared_b`, `shared_ab`).

Only built where it can differ from the default: `TN` must be a multiple of four
and hold more than one group, so of the nine tiles only the three with `TN=8`
get a second copy (42 of the 168 kernels). `--b-split off/on/both`, default
`both`; asking for `on` at a tile that has no split build is not an error, that
tile simply has the one mapping. With `off` the generated SPIR-V is
byte-identical to what it was before this axis existed, so turning the axis on
is the only thing that can move the numbers.

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

## Quick start

A prebuilt arm64-v8a binary is attached to every
[release](https://github.com/minsuk22/exynos-gemm-vulkan-bench/releases), so
there is no NDK to set up. The asset name carries the version and the URL names
the release rather than `latest`, so you always know which binary you have.

```sh
BIN=gemm_vk_bench-v0.5.0-android-arm64-v8a
curl -LO https://github.com/minsuk22/exynos-gemm-vulkan-bench/releases/download/v0.5.0/$BIN
adb push $BIN /data/local/tmp/
adb shell chmod 755 /data/local/tmp/$BIN
```

Then, in order of what you probably want:

```sh
# 1. what am I holding? version, and the 168 embedded kernels with their
#    FMA counts. Touches no GPU, so it also works as a smoke test.
adb shell /data/local/tmp/$BIN --version

# 2. does the GPU compute the right matrix? checks against a CPU reference
#    and that every kernel agrees bit-for-bit.
adb shell /data/local/tmp/$BIN --mode check --iters 2

# 3. the benchmark: A 576x160 * B 160x960 at tile 4x4 -- the four staging
#    variants, single and double buffered, so 7 kernels
adb shell /data/local/tmp/$BIN

# 4. every built tile and mapping, ranked
adb shell /data/local/tmp/$BIN --sweep --iters 50
```

Run `--mode check` at least once on a new device before trusting any number
from `--mode perf`: it is the only thing that catches a kernel that is fast
because it is computing the wrong matrix.

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

Output: `out\gemm_vk_bench-v<version>-android-arm64-v8a` — a single
self-contained arm64-v8a PIE. The SPIR-V
for all 168 kernels is embedded in the binary (no shader files to push), and
the C++ runtime is statically linked, so it only needs the platform's
`libvulkan.so` / `libc` / `libm` / `libdl`.

## Run

```powershell
.\run_on_device.ps1 -Mode perf      # timing only  (default), tile 4x4
.\run_on_device.ps1 -Mode check     # timing + result verification
.\run_on_device.ps1 -Sweep          # every built tile and mapping, ranked
```

The default shape is `A 576x160 * B 160x960`; `-Sizes` takes `MxKxN` (or a bare
number for a square case) and accepts a comma separated list.

A sweep runs 84 kernels per shape: 9 tiles x 7 staging/buffering combinations,
plus a second B mapping for the three `TN=8` tiles. The default shape is only
0.177 GFLOP, so the whole sweep is quick — but each dispatch is well under a
millisecond, which is small enough that submit overhead and clock ramping show
up as run-to-run spread. Raise `--iters` (50 or more costs nothing here) and
read the median.

The script pushes the binary to `/data/local/tmp/gemm_bench`, runs it, and
saves the console log plus a CSV under `results\`.

Manual equivalent (any OS with adb):

```sh
BIN=gemm_vk_bench-v0.5.0-android-arm64-v8a
adb push out/$BIN /data/local/tmp/
adb shell chmod 755 /data/local/tmp/$BIN
adb shell /data/local/tmp/$BIN --mode perf
```

### Asking one question at a time

A full sweep ranks everything at once, which is the wrong tool when you want to
know whether a single change pays. Each axis can be isolated — hold the rest
fixed and let the flag run both settings side by side:

```sh
D=/data/local/tmp/$BIN

# does the split B mapping pay, and where? (both settings, TN=8 tiles)
adb shell $D --tile 4x8 --tile 6x8 --tile 8x8 --b-split both

# ... narrowed to the two variants that read B straight from global memory,
#     where it moves the load addresses rather than the LDS reads
adb shell $D --tile 8x8 --b-split both --kernels none,shared_a

# does ping-ponging the LDS tiles pay?
adb shell $D --tile 8x8 --double-buffer both --kernels shared_ab

# is uploading A transposed worth it?
adb shell $D --a-layout both --kernels shared_a

# how does the answer move with the shape?
adb shell $D --sizes 576x160x960,2048,4096 --tile 8x8

# keep the numbers for later
adb shell $D --sweep --iters 50 --csv /data/local/tmp/gemm.csv
adb pull /data/local/tmp/gemm.csv
```

`--tile` is repeatable; `--kernels`, `--a-layout`, `--double-buffer` and
`--b-split` all narrow or widen the set independently. Every combination they
name is SPIR-V that was already compiled into the binary — no shader is built
at runtime, only the Vulkan pipelines for the kernels you actually selected,
which is why narrowing the set also cuts the startup cost.

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
--sizes <list>        comma separated shapes (default 576x160x960).
                      MxKxN is A MxK times B KxN; a bare S means SxSxS
--iters <n>           timed iterations per kernel (default 5)
--warmup <n>          untimed warmup iterations (default 2)
--kernels <list>      subset of none,shared_a,shared_b,shared_ab
--tile <TMxTN>        micro-tile per invocation, default 4x4. Repeatable.
--a-layout <l>        storage order of A: col (default), row, or both
--double-buffer <d>   LDS tile ping-pong: both (default), on, or off
--b-split <s>         B column mapping: both (default), on, or off
--sweep               run every built (TM,TN) pair and rank the results
--samples <n>         check mode: elements verified against CPU (default 4096)
--full-check          check mode: verify the whole matrix on CPU (slow)
--device <idx>        physical device index
--list-devices        list Vulkan devices and exit
--validation          enable VK_LAYER_KHRONOS_validation
--seed <n>            RNG seed for A and B (default 12345)
--csv <path>          append per-run results to a CSV
```

Only `K` has to be a multiple of 16 (the K tile) — the k loop walks whole
tiles. `M` and `N` are free and need not divide the block tile; see "Partial
tiles" above.

## Output

Per run you get GPU time, wall time and TFLOPS; then a per-shape summary with
best / median / mean, TFLOPS at best and mean, and speedup relative to the
first kernel in the list.

```
[shared_ab 8x8 A-col db     bs4 ]  A:shared  B:shared  block 128x128  grid 8x5
    run 1/5   gpu   XXX.XXX ms   wall   XXX.XXX ms    X.XXXX TFLOPS
    ...

 rank kernel      tile block     A / B             A-ord    dbuf bsplit   best_ms    med_ms TFLOPS_bst  vs best
    1 shared_ab    8x8 128x128   shared / shared      col      on     on   ...
    2 ...

 BEST at 576x160x960: shared_ab  tile 8x8  A col-major  double-buffer on  B-split on  ...
```

Rows are ranked by best time, so the winner is the first line and the `BEST`
line names it explicitly. Ranking is per shape — the best tile for one shape
need not be the best for another.

FLOP count is the standard `2*M*N*K`: 0.177 GFLOP for the default
576x160x960, 17.180 GFLOP at 2048³, 137.439 GFLOP at 4096³. Note that GFLOP
divided by milliseconds is already TFLOP/s
(`1e9 FLOP / 1e-3 s = 1e12 FLOP/s`) — the conversion lives in one `constexpr`
guarded by a `static_assert`, because an extra `/1e3` open-coded at five call
sites once reported every result 1000x low. Timing comes from GPU timestamp queries (`VK_QUERY_TYPE_TIMESTAMP`,
scaled by `timestampPeriod`) when the queue family supports them, falling back
to wall-clock around submit→fence otherwise; the summary states which was used.

## Notes on measurement

* Small shapes are dominated by fixed costs. The default 576x160x960 is only
  0.177 GFLOP and K is 160, i.e. ten K-tiles, so a dispatch is short enough
  that launch overhead and DVFS ramping are a real share of it. Use a large
  `--iters` and read the median; treat the ranking as more trustworthy than
  the absolute TFLOPS.
* Phones throttle. A 4096³ run moves 192 MiB of buffers and runs the GPU flat
  out; run it twice and compare if the numbers look unstable, or raise
  `--iters` and read the median rather than the mean.
* Memory: each shape allocates 3 device buffers plus one host-visible staging
  buffer — 5 MiB in total for the default shape, 4×64 MiB at 4096³. If
  allocation fails on a memory-constrained device, run the shapes separately.
* `--mode check` allocates two extra host copies of A and B (0.9 MiB for the
  default shape, 128 MiB at 4096³) for the CPU reference. Use `--mode perf`
  for the numbers you report.
