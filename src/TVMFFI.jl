#=
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
=#

"""
    TVMFFI

Julia bindings for TVM FFI (Foreign Function Interface).

This package provides a Julia interface to TVM's C API, enabling
machine learning model compilation and execution from Julia.

# Core Types
- `DLDevice`: Device abstraction (CPU, CUDA, etc.)
- `DLDataType`: Data type descriptor
- `TVMError`: Error handling
- `TVMString`: ABI-stable string type
- `TVMFunction`: Function objects
- `TVMTensor`: N-dimensional array type

# Design Philosophy
- Direct C API calls via ccall (no intermediate layer)
- Julia's GC manages object lifetimes via finalizers
- Simple, clear abstractions without over-engineering
"""
module TVMFFI

# ============================================================================
# Public API - For end users
# ============================================================================

# Core types
export DLDevice, DLDataType      # Device and data type
export TVMError                  # Error handling
export TVMFunction               # Function objects
export TVMTensor                 # N-dimensional arrays
export TVMModule                 # Compiled modules
export TensorView                # Lightweight tensor view

# Error types (for catching specific exceptions)
export TVMErrorKind
export ValueError, TypeError, RuntimeError, AttributeError, KeyError, IndexError

# Device creation
export cpu, cuda, opencl, vulkan, metal, rocm

# Tensor utilities
export shape, dtype, device      # Query tensor properties

# Function API
export get_global_func           # Get TVM global function
export register_global_func      # Register Julia function to TVM

# Module API
export load_module               # Load compiled module
export get_function              # Get function from module
export system_lib                # Get system library

# Object registration (for wrapping TVM types in Julia)
export @register_object          # Register Julia struct as TVM object
export type_index, type_key      # Query type info

# Version info
export tvm_ffi_version

# DLPack interop (self-contained, no DLPack.jl dependency)
export from_dlpack               # TVMTensor → Julia Array
export dldevice                  # Get DLDevice from array

# GPU support
export supports_gpu_backend
export list_available_gpu_backends

# ============================================================================
# Advanced API - Available but not exported (use TVMFFI.xxx)
# ============================================================================
# Low-level types: DLTensor, DLDeviceType, DLDataTypeCode, TVMObject
# String types: TVMString, TVMBytes
# Any containers: TVMAny, TVMAnyView, take_value, copy_value, raw_data
# Internal: check_call, dtype_to_julia_type
# Object registration: register_object, get_type_index, @register_object_simple
# Reflection API: get_type_info, get_fields, get_methods, FieldInfo, MethodInfo
#                 get_field_value, set_field_value!, call_method, get_method_function
#                 has_ffi_init, ffi_init
# Module introspection: write_to_file, inspect_source, get_module_kind, implements_function
# Debug: print_gpu_info, gpu_array_info

# Include submodules in dependency order
include("LibTVMFFI.jl")
include("any.jl")                 # TVMAny/TVMAnyView - ownership-aware containers
include("error.jl")
include("dtype.jl")
include("device.jl")
include("string.jl")
include("object.jl")
include("utils.jl")               # Internal utilities (_get_root_array)
include("tensor.jl")              # Defines DLTensor, TensorView and basic conversions
include("gpuarrays_support.jl")   # GPU abstraction (extends TensorView constructor)
include("dlpack.jl")              # DLPack zero-copy tensor exchange (defines _WRAPPED_ARRAYS)
include("function.jl")            # Defines TVMFunction (uses _unregister_wrapped_array)
include("module.jl")              # Defines TVMModule
include("conversion.jl")          # ABI boundary - depends on all types above

# Module initialization
function __init__()
    # Initialize module API (cache global functions)
    _init_module_api()
    # Initialize function API (callbacks)
    _init_function_api()
    # Initialize object API (type registry)
    _init_object_api()
    # Initialize DLPack API (deleter function)
    _init_dlpack_api()
end

"""
    tvm_ffi_version() -> VersionNumber

Get the TVM FFI version as a Julia `VersionNumber`.

This function queries the C API for version information and converts it
to Julia's standard version type for easy comparison and display.

# Examples
```julia
julia> v = tvm_ffi_version()
v"0.1.2"

julia> v >= v"0.1.0"
true
```
"""
function tvm_ffi_version()
    ver = LibTVMFFI.TVMFFIGetVersion()
    return VersionNumber(ver.major, ver.minor, ver.patch)
end

end # module
