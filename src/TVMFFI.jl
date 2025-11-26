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

# Export core types
export DLDevice, DLDataType, DLDeviceType, DLDataTypeCode
export DLTensor, TensorView
export TVMError, TVMErrorKind
export TVMString, TVMBytes
export TVMFunction
export TVMTensor
export TVMModule
export TVMObject

# Export Any/AnyView types (ownership-aware value containers)
export TVMAny, TVMAnyView
export take_value, copy_value, raw_data

# Export error kinds
export ValueError, TypeError, RuntimeError, AttributeError, KeyError, IndexError

# Export utility functions
export check_call, shape, dtype, device
export get_global_func, register_global_func
export register_object, get_type_index, type_index, type_key
export @register_object, @register_object_simple
export get_type_info, get_fields, get_methods
export FieldInfo, MethodInfo
export get_field_value, call_method, get_method_function
export to_julia_array
export cpu, cuda, opencl, vulkan, metal, rocm
export tvm_ffi_version, dtype_to_julia_type

# Export high-level module API
export load_module, get_function, system_lib
export write_to_file, inspect_source, get_module_kind, implements_function

# Export GPU support functions
# Note: TensorView(arr) handles both CPU and GPU arrays!
export supports_gpu_backend, list_available_gpu_backends
export print_gpu_info, gpu_array_info

# Re-export DLPack.from_dlpack for TVMTensor → Array conversion
# TVMTensor(arr) constructor is available via TVMTensor export
using DLPack: from_dlpack
export from_dlpack

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
include("function.jl")            # Defines TVMFunction
include("module.jl")              # Defines TVMModule
include("conversion.jl")          # ABI boundary - depends on all types above
include("dlpack.jl")              # DLPack zero-copy tensor exchange

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
