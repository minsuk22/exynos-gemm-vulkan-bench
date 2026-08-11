cmake_minimum_required(VERSION 3.18)
project(gemm_vk_bench VERSION 0.1.2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- SPIR-V generation -------------------------------------------------------
# Each kernel variant comes from the same shaders/gemm.comp, toggled with -D.
#
# -O is NOT optional here. glslangValidator/glslc without it emit a fully rolled
# loop nest with acc[TM][TN], regA[] and regB[] left as function-local arrays
# indexed by loop variables; drivers park those in scratch memory and every FMA
# turns into a memory round trip (measured: ~0.0008 TFLOPS). With -O the loops
# unroll, the arrays become SSA values, and the inner loop is 256 flat FMAs.
#
# glslc is used rather than glslangValidator because it runs spirv-opt itself
# and ships inside the NDK, so no separate glslang install is needed.

find_program(GLSLC
  NAMES glslc glslc.exe
  HINTS ${GLSLC_DIR} $ENV{GLSLC_DIR}
        ${ANDROID_NDK}/shader-tools/${ANDROID_HOST_TAG}
        ${ANDROID_NDK}/shader-tools
        ENV VULKAN_SDK
  PATH_SUFFIXES bin windows-x86_64 linux-x86_64 darwin-x86_64)
if(NOT GLSLC)
  message(FATAL_ERROR "glslc not found. It ships in the NDK under "
                      "shader-tools/<host>/; pass -DGLSLC=<path> to override.")
endif()
message(STATUS "glslc: ${GLSLC}")

set(GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/gen)
file(MAKE_DIRECTORY ${GEN_DIR})
set(SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/shaders/gemm.comp)
set(SPV_INCS "")

#            variant name   USE_SHARED_A  USE_SHARED_B
set(VARIANTS "none:0:0" "shared_a:1:0" "shared_b:0:1" "shared_ab:1:1")

foreach(v IN LISTS VARIANTS)
  string(REPLACE ":" ";" parts ${v})
  list(GET parts 0 vname)
  list(GET parts 1 usea)
  list(GET parts 2 useb)
  set(inc ${GEN_DIR}/gemm_${vname}.inc)
  # -mfmt=c emits a braced uint32 initializer list, included directly by main.cpp
  add_custom_command(
    OUTPUT  ${inc}
    COMMAND ${GLSLC} -O --target-env=vulkan1.1 -fshader-stage=compute
            -DUSE_SHARED_A=${usea} -DUSE_SHARED_B=${useb}
            -mfmt=c -o ${inc} ${SHADER_SRC}
    DEPENDS ${SHADER_SRC}
    COMMENT "glslc -O gemm.comp -> gemm_${vname}.inc (A_shared=${usea} B_shared=${useb})"
    VERBATIM)
  list(APPEND SPV_INCS ${inc})
endforeach()

add_custom_target(gemm_shaders DEPENDS ${SPV_INCS})

# --- executable --------------------------------------------------------------
add_executable(gemm_vk_bench src/main.cpp)
add_dependencies(gemm_vk_bench gemm_shaders)
target_include_directories(gemm_vk_bench PRIVATE ${GEN_DIR})
target_compile_options(gemm_vk_bench PRIVATE -O3 -Wall)

# The version is baked in and printed in the report header, so a log always
# identifies the binary that produced it.
target_compile_definitions(gemm_vk_bench PRIVATE
  GEMM_BENCH_VERSION="v${PROJECT_VERSION}")

# Carry the version in the file name too, so a stale push to the device is
# visible before it is ever run.
set_target_properties(gemm_vk_bench PROPERTIES
  OUTPUT_NAME "gemm_vk_bench-v${PROJECT_VERSION}")

if(ANDROID)
  # libvulkan.so is part of the platform from API 24 on; the NDK ships the stub.
  target_link_libraries(gemm_vk_bench vulkan)
  # -static-libstdc++ / c++_static (set via ANDROID_STL) keeps the binary
  # self-contained so it can just be pushed to /data/local/tmp and run.
else()
  find_package(Vulkan REQUIRED)
  target_link_libraries(gemm_vk_bench Vulkan::Vulkan)
endif()
find_package(Threads REQUIRED)
target_link_libraries(gemm_vk_bench Threads::Threads)
