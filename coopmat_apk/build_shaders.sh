#!/usr/bin/env bash
# Compile shaders/coopmat_gemm.comp once per (A, B, C) type combination into
# app/src/main/assets/shaders/gemm_<A>_<B>_<C>[_sat]_<v4|v1>.spv
#
# The app looks a kernel up by the names of the component types the device
# advertises, so a combo that is not built here shows up as "no kernel" in the
# results instead of failing. Add a line to COMBOS to cover a new one.
#
# Needs glslang >= 15 for GL_EXT_bfloat16 / GL_EXT_float_e4m3 / _e5m2.
#   GLSLANG=/path/to/glslangValidator ./build_shaders.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
GLSLANG="${GLSLANG:-/d/Android/glslang-16.6.0/bin/glslang.exe}"
# Optional: a spirv-val new enough to know SPV_KHR_cooperative_matrix (the NDK one is not).
SPIRV_VAL="${SPIRV_VAL:-}"
SRC="$HERE/shaders/coopmat_gemm.comp"
OUT="$HERE/app/src/main/assets/shaders"
TENV="spirv1.3"

# short-name : GLSL type : bytes : extension flag
declare -A GLSL=( [f16]=float16_t [f32]=float32_t [bf16]=bfloat16_t
                  [e4m3]=floate4m3_t [e5m2]=floate5m2_t
                  [s8]=int8_t [u8]=uint8_t [s16]=int16_t [u16]=uint16_t
                  [s32]=int32_t [u32]=uint32_t )
declare -A BYTES=( [f16]=2 [f32]=4 [bf16]=2 [e4m3]=1 [e5m2]=1
                   [s8]=1 [u8]=1 [s16]=2 [u16]=2 [s32]=4 [u32]=4 )

#        A     B     C     sat-variants
COMBOS=(
  "f16  f16  f16  0"
  "f16  f16  f32  0"
  "f32  f32  f32  0"
  "bf16 bf16 bf16 0"
  "bf16 bf16 f32  0"
  "e4m3 e4m3 f16  0"
  "e4m3 e4m3 f32  0"
  "e5m2 e5m2 f16  0"
  "e5m2 e5m2 f32  0"
  "e4m3 e5m2 f16  0"
  "e4m3 e5m2 f32  0"
  "e5m2 e4m3 f16  0"
  "e5m2 e4m3 f32  0"
  "s8   s8   s32  01"
  "u8   u8   u32  01"
  "u8   u8   s32  01"
  "s8   u8   s32  01"
  "u8   s8   s32  01"
  "s8   s8   s16  01"
  "u8   u8   u16  01"
)

mkdir -p "$OUT"
rm -f "$OUT"/gemm_*.spv
n=0
for row in "${COMBOS[@]}"; do
  read -r A B C SATS <<<"$row"
  flags=()
  for t in "$A" "$B" "$C"; do
    case "$t" in bf16) flags+=(-DNEED_BF16=1);; e4m3|e5m2) flags+=(-DNEED_FP8=1);; esac
  done
  for (( i=0; i<${#SATS}; i++ )); do
    sat="${SATS:$i:1}"
    suffix=""; [[ "$sat" == 1 ]] && suffix="_sat"
    for ld in v4 v1; do
      if [[ $ld == v4 ]]; then ldt=uvec4; ldb=16; else ldt=uint; ldb=4; fi
      name="gemm_${A}_${B}_${C}${suffix}_${ld}.spv"
      "$GLSLANG" -V --target-env "$TENV" -Os \
        -DA_T="${GLSL[$A]}" -DB_T="${GLSL[$B]}" -DC_T="${GLSL[$C]}" \
        -DA_BYTES="${BYTES[$A]}" -DB_BYTES="${BYTES[$B]}" \
        -DLD_T="$ldt" -DLD_BYTES="$ldb" -DSAT="$sat" "${flags[@]}" \
        "$SRC" -o "$OUT/$name" >/dev/null
      if [[ -n "$SPIRV_VAL" ]]; then
        "$SPIRV_VAL" --target-env vulkan1.3 "$OUT/$name"
      fi
      n=$((n+1))
    done
  done
done
echo "built $n kernels into $OUT"
