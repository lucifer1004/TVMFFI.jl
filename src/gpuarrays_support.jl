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

Supports multiple GPU backends via package extensions:
- CUDA.jl (NVIDIA)
- AMDGPU.jl (AMD ROCm)
- Metal.jl (Apple)
- oneAPI.jl (Intel)

No dependency on GPUArrays.jl or DLPack.jl - self-contained implementation.

# Design Philosophy
- Default `dldevice(arr)` returns CPU device
- GPU extensions override `dldevice` for their array types
- Type dispatch via Julia's multiple dispatch

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

using .LibTVMFFI

# ============================================================================
# Device Detection API
# ============================================================================

"""
    dldevice(arr) -> DLDevice

Get the DLPack device for an array.

Default implementation returns CPU device. GPU extensions override this
for their specific array types (CuArray, MtlArray, ROCArray, etc.).

# Returns
- `DLDevice(kDLCPU, 0)` for CPU arrays (default)
- `DLDevice(kDLCUDA, device_id)` for CUDA arrays
- `DLDevice(kDLMetal, device_id)` for Metal arrays
- `DLDevice(kDLROCM, device_id)` for ROCm arrays
"""
dldevice(::AbstractArray) = DLDevice(Int32(LibTVMFFI.kDLCPU), Int32(0))

"""
    _dldevice_type_to_symbol(device_type) -> Symbol

Convert DLDeviceType to a human-readable symbol for printing.
"""
function _dldevice_type_to_symbol(device_type::Integer)
    type_map = Dict(
        Int32(LibTVMFFI.kDLCPU) => :CPU,
        Int32(LibTVMFFI.kDLCUDA) => :CUDA,
        Int32(LibTVMFFI.kDLCUDAHost) => :CUDAHost,
        Int32(LibTVMFFI.kDLCUDAManaged) => :CUDAManaged,
        Int32(LibTVMFFI.kDLROCM) => :ROCm,
        Int32(LibTVMFFI.kDLROCMHost) => :ROCmHost,
        Int32(LibTVMFFI.kDLMetal) => :Metal,
        Int32(LibTVMFFI.kDLVulkan) => :Vulkan,
        Int32(LibTVMFFI.kDLOpenCL) => :OpenCL,
        Int32(LibTVMFFI.kDLOneAPI) => :oneAPI
    )
    return get(type_map, Int32(device_type), :Unknown)
end

# ============================================================================
# GPU Backend Detection
# ============================================================================

"""
    supports_gpu_backend(backend::Symbol) -> Bool

Check if a specific GPU backend is available.

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
        println("    • Backend available (device enumeration requires importing package)")
    end
end

# ============================================================================
# TensorView Constructor for Generic AbstractArray
# ============================================================================

"""
    TensorView(arr::AbstractArray)

Construct a TensorView from any AbstractArray, including GPU arrays.

# Design Philosophy
ONE constructor, ONE type, ALL devices.
- Use dldevice() for device detection (extensible via package extensions)
- Create appropriate device automatically
- Return unified TensorView

# Performance
Shape and strides are stored as NTuple (inline), avoiding heap allocation.
This saves ~144 bytes per TensorView creation.

# Examples
```julia
using CUDA

# CPU and GPU - same API!
cpu_view = TensorView(cpu_arr)     # Auto: CPU device
gpu_view = TensorView(gpu_arr)     # Auto: CUDA device

# Both return TensorView{T, S, N}
# Device info is in view.dltensor.device
```
"""
function TensorView(arr::S) where {S <: AbstractArray}
    T = eltype(arr)
    N = ndims(arr)

    # Auto-detect device using dldevice (handles all GPU backends!)
    device = dldevice(arr)

    # Get shape and strides as tuples (zero allocation)
    shape_tuple = NTuple{N, Int64}(size(arr))
    strides_tuple = NTuple{N, Int64}(Base.strides(arr))

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
    if device.device_type == Int32(LibTVMFFI.kDLMetal)
        arr_type = typeof(arr)
        if hasfield(arr_type, :offset)
            base_offset = arr.offset
            byte_offset = UInt64(base_offset) * element_size
        else
            root_arr = _get_root_array(arr)
            if arr !== root_arr
                # Calculate SubArray offset relative to root array
                root_strides = Base.strides(root_arr)
                first_indices = ntuple(i -> first(arr.indices[i]), N)

                # Calculate offset: sum((first_index - 1) * stride * element_size)
                subarray_offset = sum(
                    (first_indices[i] - 1) * root_strides[i] * element_size
                    for i in 1:N
                )
                byte_offset = UInt64(subarray_offset)
            end
        end
    end

    # Convert pointer to Ptr{Cvoid} - handle both regular Ptr and GPU pointers (CuPtr, etc.)
    # GPU pointers need conversion through UInt since they don't directly convert to Ptr{Cvoid}
    data_ptr = Ptr{Cvoid}(UInt(arr_ptr))

    return TensorView{T, S, N}(
        data_ptr,
        device,
        dt,
        shape_tuple,
        strides_tuple,
        byte_offset,
        arr,
        JuliaOwned
    )
end

# ============================================================================
# GPU Array Info
# ============================================================================

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
        # Get device info
        device = dldevice(arr)
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
