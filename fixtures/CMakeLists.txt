# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

cmake_minimum_required(VERSION 3.20)
project(tvmffi_test_fixtures LANGUAGES CXX)

# Optional CUDA support
option(BUILD_CUDA_FIXTURES "Build CUDA test fixtures" ON)
if(BUILD_CUDA_FIXTURES)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_AVAILABLE TRUE)
    message(STATUS "CUDA support enabled")
  else()
    set(CUDA_AVAILABLE FALSE)
    message(STATUS "CUDA not found, skipping CUDA fixtures")
  endif()
else()
  set(CUDA_AVAILABLE FALSE)
  message(STATUS "CUDA fixtures disabled (BUILD_CUDA_FIXTURES=OFF)")
endif()

# ============================================================================
# Find Julia and get TVMFFI_jll paths
# ============================================================================

# Find Julia executable
find_program(JULIA_EXECUTABLE julia REQUIRED)
message(STATUS "Found Julia: ${JULIA_EXECUTABLE}")

# Get TVMFFI_jll artifact directory by running Julia
execute_process(
  COMMAND "${JULIA_EXECUTABLE}" --startup-file=no --project=.. -e 
    "using TVMFFI_jll; print(TVMFFI_jll.artifact_dir)"
  OUTPUT_VARIABLE TVMFFI_ARTIFACT_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE JULIA_RESULT
)

if(NOT JULIA_RESULT EQUAL 0)
  message(FATAL_ERROR 
    "Failed to get TVMFFI_jll artifact directory. "
    "Make sure TVMFFI.jl is installed: julia -e 'using Pkg; Pkg.add(\"TVMFFI\")'"
  )
endif()

message(STATUS "TVMFFI artifact dir: ${TVMFFI_ARTIFACT_DIR}")

# Construct paths from artifact directory
set(TVMFFI_INCLUDE_DIR "${TVMFFI_ARTIFACT_DIR}/include")
if(WIN32)
set(TVMFFI_LIB_DIR "${TVMFFI_ARTIFACT_DIR}/bin")
else()
set(TVMFFI_LIB_DIR "${TVMFFI_ARTIFACT_DIR}/lib")
endif()

# Determine shared library name based on platform
if(WIN32)
  set(TVMFFI_LIB_NAME "libtvm_ffi.dll")
elseif(APPLE)
  set(TVMFFI_LIB_NAME "libtvm_ffi.dylib")
else()
  set(TVMFFI_LIB_NAME "libtvm_ffi.so")
endif()

set(TVMFFI_SHARED_LIB "${TVMFFI_LIB_DIR}/${TVMFFI_LIB_NAME}")

message(STATUS "TVMFFI Include: ${TVMFFI_INCLUDE_DIR}")
message(STATUS "TVMFFI Lib Dir: ${TVMFFI_LIB_DIR}")
message(STATUS "TVMFFI Shared: ${TVMFFI_SHARED_LIB}")

# Verify paths exist
if(NOT EXISTS "${TVMFFI_INCLUDE_DIR}")
  message(FATAL_ERROR "TVMFFI include directory not found: ${TVMFFI_INCLUDE_DIR}")
endif()

if(NOT EXISTS "${TVMFFI_SHARED_LIB}")
  message(FATAL_ERROR "TVMFFI shared library not found: ${TVMFFI_SHARED_LIB}")
endif()

# ============================================================================
# Build fixtures
# ============================================================================

# Build add_one_cpu fixture
add_library(add_one_cpu SHARED add_one_cpu.cc)

# Add include directories
target_include_directories(add_one_cpu PRIVATE ${TVMFFI_INCLUDE_DIR})

# Link against TVMFFI shared library
target_link_libraries(add_one_cpu PRIVATE ${TVMFFI_SHARED_LIB})

# Set output properties
set_target_properties(
  add_one_cpu
  PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/"
    PREFIX ""
)

# Set correct suffix based on platform
if(WIN32)
  set_target_properties(add_one_cpu PROPERTIES SUFFIX ".dll")
elseif(APPLE)
  set_target_properties(add_one_cpu PROPERTIES SUFFIX ".dylib")
else()
  set_target_properties(add_one_cpu PROPERTIES SUFFIX ".so")
endif()

# On Linux/macOS, set RPATH to find libtvm_ffi_shared.so at runtime
if(UNIX)
  set_target_properties(add_one_cpu PROPERTIES
    INSTALL_RPATH "${TVMFFI_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
endif()

message(STATUS "Configured fixture: add_one_cpu")

# ============================================================================
# Optional CUDA fixtures
# ============================================================================

if(CUDA_AVAILABLE)
  # Detect CUDA architectures if not specified by user
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    # Try to detect from nvidia-smi
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
      execute_process(
        COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE DETECTED_CUDA_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE SMI_RESULT
      )
      if(SMI_RESULT EQUAL 0 AND NOT DETECTED_CUDA_ARCH STREQUAL "")
        # Remove dots from compute capability (e.g., "8.0" -> "80")
        string(REPLACE "." "" DETECTED_CUDA_ARCH "${DETECTED_CUDA_ARCH}")
        # Handle multiple GPUs (take first line)
        string(REPLACE "\n" ";" DETECTED_CUDA_ARCH "${DETECTED_CUDA_ARCH}")
        list(GET DETECTED_CUDA_ARCH 0 DETECTED_CUDA_ARCH)
        set(CMAKE_CUDA_ARCHITECTURES "${DETECTED_CUDA_ARCH}")
        message(STATUS "Detected CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")
      endif()
    endif()
    
    # Fallback: compile for common architectures (Volta to Hopper)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
      set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
      message(STATUS "Using default CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
  else()
    message(STATUS "Using user-specified CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  endif()
  
  # Build add_one_cuda fixture
  add_library(add_one_cuda SHARED add_one_cuda.cu)
  
  # Add include directories
  target_include_directories(add_one_cuda PRIVATE ${TVMFFI_INCLUDE_DIR})
  
  # Link against TVMFFI shared library
  target_link_libraries(add_one_cuda PRIVATE ${TVMFFI_SHARED_LIB})
  
  # Set CUDA properties
  set_target_properties(add_one_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/"
    PREFIX ""
  )
  
  # Set correct suffix based on platform
  if(WIN32)
    set_target_properties(add_one_cuda PROPERTIES SUFFIX ".dll")
  elseif(APPLE)
    set_target_properties(add_one_cuda PROPERTIES SUFFIX ".dylib")
  else()
    set_target_properties(add_one_cuda PROPERTIES SUFFIX ".so")
  endif()
  
  # On Linux/macOS, set RPATH
  if(UNIX)
    set_target_properties(add_one_cuda PROPERTIES
      INSTALL_RPATH "${TVMFFI_LIB_DIR}"
      BUILD_WITH_INSTALL_RPATH TRUE
    )
  endif()
  
  message(STATUS "Configured fixture: add_one_cuda")
endif()

