cmake_minimum_required(VERSION 3.21)
project(kernelpilot LANGUAGES CXX)

# Kernels (.hip) are C++17. bindings.cpp includes PyTorch headers, and recent
# PyTorch releases require C++20 for those, so host C++ is compiled as C++20.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# KP_BACKEND: "rocm" or "cuda". The Makefile picks it from torch.version.hip.
set(KP_BACKEND "cuda" CACHE STRING "GPU backend: rocm or cuda")

file(GLOB KP_HIP_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/*.hip)

if(KP_BACKEND STREQUAL "rocm")
  enable_language(HIP)
  set(CMAKE_HIP_STANDARD 17)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  set(KP_KERNEL_SOURCES ${KP_HIP_SOURCES})
  set_source_files_properties(${KP_KERNEL_SOURCES} PROPERTIES LANGUAGE HIP)
elseif(KP_BACKEND STREQUAL "cuda")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
      set(CMAKE_CUDA_ARCHITECTURES native)
    else()
      set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # nvcc keys off the file extension, so compile a .cu copy of each HIP file.
  # compat.h maps the few HIP-specific names to CUDA.
  set(KP_KERNEL_SOURCES "")
  foreach(src ${KP_HIP_SOURCES})
    get_filename_component(stem ${src} NAME_WE)
    set(dst ${CMAKE_CURRENT_BINARY_DIR}/cu/kernels/${stem}.cu)
    configure_file(${src} ${dst} COPYONLY)
    list(APPEND KP_KERNEL_SOURCES ${dst})
  endforeach()
  configure_file(csrc/compat.h ${CMAKE_CURRENT_BINARY_DIR}/cu/compat.h COPYONLY)
  configure_file(csrc/launchers.h ${CMAKE_CURRENT_BINARY_DIR}/cu/launchers.h COPYONLY)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
else()
  message(FATAL_ERROR "KP_BACKEND must be rocm or cuda, got ${KP_BACKEND}")
endif()

find_package(Torch REQUIRED)

add_library(kernelpilot_ops SHARED csrc/bindings.cpp ${KP_KERNEL_SOURCES})
target_include_directories(kernelpilot_ops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
target_link_libraries(kernelpilot_ops PRIVATE ${TORCH_LIBRARIES})
target_compile_options(kernelpilot_ops PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
if(KP_BACKEND STREQUAL "rocm")
  target_compile_options(kernelpilot_ops PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3>)
endif()
