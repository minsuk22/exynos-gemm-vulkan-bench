cmake_minimum_required(VERSION 3.18)
project(gemm_vk_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- SPIR-V generation -------------------------------------------------------
# Each kernel variant comes from the same shaders/gemm.comp, toggled with -D.
# glslangValidator emits a C header (`const uint32_t <name>_spv[]`) so the
# shaders end up embedded in the executable and we ship a single file.

find_program(GLSLANG_VALIDATOR
  NAMES glslangValidator glslangValidator.exe
  HINTS ${GLSLANG_DIR} $ENV{GLSLANG_DIR} ENV VULKAN_SDK
  PATH_SUFFIXES bin Bin)
if(NOT GLSLANG_VALIDATOR)
  message(FATAL_ERROR "glslangValidator not found. Pass -DGLSLANG_VALIDATOR=<path>.")
endif()
message(STATUS "glslangValidator: ${GLSLANG_VALIDATOR}")

set(GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/gen)
file(MAKE_DIRECTORY ${GEN_DIR})
set(SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/shaders/gemm.comp)
set(SPV_HEADERS "")

#            variant name   USE_SHARED_A  USE_SHARED_B
set(VARIANTS "none:0:0" "shared_a:1:0" "shared_b:0:1" "shared_ab:1:1")

foreach(v IN LISTS VARIANTS)
  string(REPLACE ":" ";" parts ${v})
  list(GET parts 0 vname)
  list(GET parts 1 usea)
  list(GET parts 2 useb)
  set(hdr ${GEN_DIR}/gemm_${vname}.h)
  add_custom_command(
    OUTPUT  ${hdr}
    COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.1
            -DUSE_SHARED_A=${usea} -DUSE_SHARED_B=${useb}
            --vn gemm_${vname}_spv -o ${hdr} ${SHADER_SRC}
    DEPENDS ${SHADER_SRC}
    COMMENT "glslangValidator gemm.comp -> gemm_${vname}.h (A_shared=${usea} B_shared=${useb})"
    VERBATIM)
  list(APPEND SPV_HEADERS ${hdr})
endforeach()

add_custom_target(gemm_shaders DEPENDS ${SPV_HEADERS})

# --- executable --------------------------------------------------------------
add_executable(gemm_vk_bench src/main.cpp)
add_dependencies(gemm_vk_bench gemm_shaders)
target_include_directories(gemm_vk_bench PRIVATE ${GEN_DIR})
target_compile_options(gemm_vk_bench PRIVATE -O3 -Wall)

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
