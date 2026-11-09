cmake_minimum_required(VERSION 3.22.1)
project(coopmatprobe CXX)

add_library(coopmat_probe SHARED coopmat_probe.cpp)
target_compile_options(coopmat_probe PRIVATE -Wall -O2)
target_link_libraries(coopmat_probe vulkan android log)
