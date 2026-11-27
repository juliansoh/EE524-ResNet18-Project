cmake_minimum_required(VERSION 3.18)

# ---------------------------------------------------------------------------------------------------------
# ResNet-18 inference engine (Au26 GPU Compute final project starter)
#
#   GPU build (default):      cmake -B build -DCMAKE_BUILD_TYPE=Release [-DCMAKE_CUDA_ARCHITECTURES=75]
#                             cmake --build build -j
#   Add the fp16 engine:      -DENABLE_FP16=ON          (Tier 4; instantiates every kernel for half)
#   No-GPU test build:        -DCPU_EMULATION=ON        (compiles the .cu files as plain C++; very slow;
#                                                        for exercising the toolchain, checker and grader)
# The leaderboard builds with the default options for sm_75 only. Add -DCMAKE_CUDA_ARCHITECTURES="75;89" to
# also target a 4090/L4 for development, "75;86" for a 3080 Ti, "75;90" for an H100/H200, or =native (CMake >= 3.24).
# Keep 75 in the list: instructions the T4 lacks (cp.async, bf16, m16n8k16 fp16 MMA, wgmma) fail at ptxas on the grader.
# CUDA 13.x targets compute capability 7.5 and newer; CUDA 12.x also builds for 5.0-7.0.
# ---------------------------------------------------------------------------------------------------------
option(CPU_EMULATION "Build the CUDA sources as C++ and run kernels serially on the CPU (no GPU needed)" OFF)
option(ENABLE_FP16   "Instantiate the half-precision engine and kernels (Tier 4)" OFF)

if(CPU_EMULATION)
  project(resnet18_engine LANGUAGES CXX)
else()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()
  project(resnet18_engine LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

file(GLOB CU_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cu)
set(CPP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/manifest.cpp)

add_executable(engine ${CPP_SOURCES} ${CU_SOURCES})
target_include_directories(engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

if(ENABLE_FP16)
  target_compile_definitions(engine PRIVATE ENABLE_FP16=1)
endif()

if(CPU_EMULATION)
  set_source_files_properties(${CU_SOURCES} PROPERTIES LANGUAGE CXX)
  target_compile_definitions(engine PRIVATE CPU_EMULATION=1)
  if(MSVC)
    target_compile_options(engine PRIVATE /TP /O2)
  else()
    # tell gcc/clang that the .cu files are C++
    set_source_files_properties(${CU_SOURCES} PROPERTIES COMPILE_OPTIONS "-x;c++")
    target_compile_options(engine PRIVATE -O2)
  endif()
else()
  set(CMAKE_CUDA_STANDARD 17)
  set_target_properties(engine PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  # -lineinfo keeps source correlation for Nsight Compute without disabling optimizations
  target_compile_options(engine PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>)
  if(NOT MSVC)
    target_compile_options(engine PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O2>)
  endif()
endif()

install(TARGETS engine RUNTIME DESTINATION bin)
