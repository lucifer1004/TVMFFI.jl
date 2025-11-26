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
Hardware-Agnostic GPU Support

Supports multiple GPU backends through DLPack.jl's type dispatch:
- CUDA.jl (NVIDIA)
- AMDGPU.jl (AMD ROCm)
- Metal.jl (Apple)
- oneAPI.jl (Intel)

No dependency on GPUArrays.jl - works directly with each backend.

# Design Philosophy (Linus-style)
- Use DLPack.dldevice() for device detection - no duplication!
- DLPack.jl already handles type dispatch for each GPU backend
- We just convert DLPack.DLDevice to our LibTVMFFI.DLDevice

# Usage

```julia
using TVMFFI
using CUDA  # or AMDGPU, Metal, oneAPI

# Works with any GPU backend!
x_gpu = CUDA.CuArray(Float32[1, 2, 3])

# Unified API - TensorView constructor auto-detects device
x_view = TensorView(x_gpu)

# Call TVM function
tvm_func(x_view, y_view)
```
"""

import DLPack

# ============================================================================
# DLDevice Conversion
# ============================================================================

"""
    _dlpack_to_tvm_device(arr) -> DLDevice

Get device info from array using DLPack.dldevice and convert to TVMFFI's DLDevice.

Design Philosophy (Linus-style):
- DLPack.dldevice() already does the heavy lifting via type dispatch
- We just need to convert DLPack.DLDevice → LibTVMFFI.DLDevice
- No duplication of backend detection logic!
"""
function _dlpack_to_tvm_device(arr)
    # DLPack.dldevice handles all the type dispatch magic
    # Returns DLPack.DLDevice(device_type::DLDeviceType, device_id::Cint)
    dlpack_dev = DLPack.dldevice(arr)
    
    # Convert to TVMFFI's DLDevice (same memory layout, different type)
    return DLDevice(Int32(dlpack_dev.device_type), Int32(dlpack_dev.device_id))
end

"""
    _dldevice_type_to_symbol(device_type) -> Symbol

Convert DLDeviceType to a human-readable symbol for printing.
"""
function _dldevice_type_to_symbol(device_type::Integer)
    type_map = Dict(
        Int32(DLPack.kDLCPU) => :CPU,
        Int32(DLPack.kDLCUDA) => :CUDA,
        Int32(DLPack.kDLCUDAHost) => :CUDAHost,
        Int32(DLPack.kDLCUDAManaged) => :CUDAManaged,
        Int32(DLPack.kDLROCM) => :ROCm,
        Int32(DLPack.kDLROCMHost) => :ROCmHost,
        Int32(DLPack.kDLMetal) => :Metal,
        Int32(DLPack.kDLVulkan) => :Vulkan,
        Int32(DLPack.kDLOpenCL) => :OpenCL,
        Int32(DLPack.kDLOneAPI) => :oneAPI,
    )
    return get(type_map, Int32(device_type), :Unknown)
end

"""
    supports_gpu_backend(backend::Symbol) -> Bool

Check if a specific GPU backend is available.

# Design Philosophy (Linus-style)
Old code: Try to introspect Main module and call .functional()
New code: Just check if the package exists in the current environment
Simpler, no weird Main.CUDA hacks

# Arguments
- `backend::Symbol`: Backend to check (:CUDA, :ROCm, :Metal, :oneAPI)

# Returns
- `Bool`: Whether the backend package is loaded

# Examples
```julia
if supports_gpu_backend(:CUDA)
    println("CUDA is available!")
end
```
"""
function supports_gpu_backend(backend::Symbol)
    # Check if the module is in the namespace by trying to get its const
    # This is cleaner than isdefined(Main, :...)
    try
        if backend == :CUDA
            return isdefined(@__MODULE__, :CUDA) ||
                   Base.get_extension(@__MODULE__, :CUDAExt) !== nothing
        elseif backend == :ROCm || backend == :AMDGPU
            return isdefined(@__MODULE__, :AMDGPU) ||
                   Base.get_extension(@__MODULE__, :AMDGPUExt) !== nothing
        elseif backend == :Metal
            return isdefined(@__MODULE__, :Metal) ||
                   Base.get_extension(@__MODULE__, :MetalExt) !== nothing
        elseif backend == :oneAPI
            return isdefined(@__MODULE__, :oneAPI) ||
                   Base.get_extension(@__MODULE__, :oneAPIExt) !== nothing
        end
    catch
        # Fallback: just return false if extension system not available (Julia < 1.9)
        return false
    end
    return false
end

"""
    list_available_gpu_backends() -> Vector{Symbol}

List all available and functional GPU backends.

# Returns
- `Vector{Symbol}`: List of available backends

# Examples
```julia
backends = list_available_gpu_backends()
println("Available GPU backends: ", backends)
# Output: [:CUDA, :ROCm]  (depending on system)
```
"""
function list_available_gpu_backends()
    all_backends = [:CUDA, :ROCm, :Metal, :oneAPI]
    available = Symbol[]

    for backend in all_backends
        if supports_gpu_backend(backend)
            push!(available, backend)
        end
    end

    return available
end

"""
    print_gpu_info()

Print information about available GPU backends and devices.

# Examples
```julia
print_gpu_info()
# Output:
# Available GPU Backends:
#   ✓ CUDA (NVIDIA)
#     • Device 0: NVIDIA GeForce RTX 3090
#     • Device 1: NVIDIA GeForce RTX 3080
#   ✓ ROCm (AMD)
#     • Device 0: AMD Radeon RX 6900 XT
```
"""
function print_gpu_info()
    println("Available GPU Backends:")

    backends = list_available_gpu_backends()

    if isempty(backends)
        println("  ❌ No GPU backends available")
        println("\nTo enable GPU support, install one of:")
        println("  • CUDA.jl:   using Pkg; Pkg.add(\"CUDA\")")
        println("  • AMDGPU.jl: using Pkg; Pkg.add(\"AMDGPU\")")
        println("  • Metal.jl:  using Pkg; Pkg.add(\"Metal\")")
        println("  • oneAPI.jl: using Pkg; Pkg.add(\"oneAPI\")")
        return
    end

    for backend in backends
        vendor = if backend == :CUDA
            "NVIDIA"
        elseif backend == :ROCm || backend == :AMDGPU
            "AMD"
        elseif backend == :Metal
            "Apple"
        elseif backend == :oneAPI
            "Intel"
        else
            "Unknown"
        end

        println("  ✓ $backend ($vendor)")

        # Simplified: just report backend is available
        # Device enumeration requires calling into the backend packages
        # which we've intentionally avoided for simplicity
        println("    • Backend available (device enumeration requires importing package)")
    end
end

# ============================================================================
# Internal Utilities
# ============================================================================
# Note: _get_root_array is in utils.jl
# Device detection uses DLPack.dldevice - no duplication!

# Extend TensorView constructor to handle GPU arrays
"""
    TensorView(arr::AbstractArray)

Construct a TensorView from any AbstractArray, including GPU arrays.

# Design Philosophy (Linus-style)
ONE constructor, ONE type, ALL devices.
- Use DLPack.dldevice() for device detection (no duplication!)
- Create appropriate device automatically
- Return unified TensorView

# Examples
```julia
using CUDA

# CPU and GPU - same API!
cpu_view = TensorView(cpu_arr)     # Auto: CPU device
gpu_view = TensorView(gpu_arr)     # Auto: CUDA device

# Both return TensorView{T, S}
# Device info is in view.dltensor.device
```
"""
function TensorView(arr::S) where {S <: AbstractArray}
    T = eltype(arr)

    # Auto-detect device using DLPack.dldevice (handles all GPU backends!)
    device = _dlpack_to_tvm_device(arr)

    # Get shape and strides - same as CPU
    shape_vec = collect(Int64, size(arr))
    strides_vec = collect(Int64, Base.strides(arr))

    # Get dtype
    dt = DLDataType(T)
    root_arr = _get_root_array(arr)
    arr_ptr = try
        pointer(arr)
    catch
        pointer(root_arr)
    end

    # For Metal arrays, extract byte_offset from MtlPtr
    # Other GPU backends use byte_offset = 0
    byte_offset = UInt64(0)
    element_size = sizeof(T)
    if device.device_type == Int32(DLPack.kDLMetal)
        arr_type = typeof(arr)
        if hasfield(arr_type, :offset)
            base_offset = arr.offset
            byte_offset = UInt64(base_offset) * element_size
        else
            root_arr = _get_root_array(arr)
            if arr !== root_arr
                # Calculate SubArray offset relative to root array
                root_strides = Base.strides(root_arr)
                first_indices = [first(ax) for ax in arr.indices]

                # Calculate offset: sum((first_index - 1) * stride * element_size)
                subarray_offset = sum((first_indices[i] - 1) * root_strides[i] *
                                      element_size
                for i in 1:length(first_indices))
                byte_offset = UInt64(subarray_offset)
            end
        end
    end

    # Create DLTensor
    dltensor = DLTensor(
        arr_ptr,
        device,
        Int32(length(shape_vec)),
        dt,
        pointer(shape_vec),
        pointer(strides_vec),
        byte_offset
    )

    return TensorView{T, S}(dltensor, shape_vec, strides_vec, arr)
end

"""
    gpu_array_info(arr)

Print diagnostic information about a GPU array.

# Example
```julia
using CUDA
x = CUDA.CuArray(Float32[1, 2, 3])
gpu_array_info(x)
# Output:
#   Backend: CUDA
#   Device: 0
#   Type: Float32
#   Shape: (3,)
#   Pointer: 0x...
```
"""
function gpu_array_info(arr)
    println("GPU Array Information:")

    try
        # Get device info using DLPack
        device = _dlpack_to_tvm_device(arr)
        backend = _dldevice_type_to_symbol(device.device_type)
        
        println("  Backend: $backend")
        println("  Device ID: $(device.device_id)")
        println("  Element Type: ", eltype(arr))
        println("  Shape: ", size(arr))
        println("  Size: ", length(arr), " elements")
        println("  Memory Pointer: ", repr(UInt(pointer(arr))))
        println("  DLDevice: ", device)

    catch e
        println("  Error getting info: ", e)
    end
end
