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

Supports multiple GPU backends through type dispatch:
- CUDA.jl (NVIDIA)
- AMDGPU.jl (AMD ROCm)
- Metal.jl (Apple)
- oneAPI.jl (Intel)

No dependency on GPUArrays.jl - works directly with each backend.

# Usage

```julia
using TVMFFI
using CUDA  # or AMDGPU, Metal, oneAPI

# Works with any GPU backend!
x_gpu = CUDA.CuArray(Float32[1, 2, 3])

# Unified API - from_julia_array auto-detects device
x_holder = from_julia_array(x_gpu)

# Call TVM function
tvm_func(x_holder, y_holder)
```
"""

"""
    gpu_backend_to_dldevice(backend::Symbol) -> DLDeviceType

Map GPU backend name to DLDeviceType.

# Arguments
- `backend::Symbol`: GPU backend (:CUDA, :ROCm, :Metal, :oneAPI)

# Returns
- `DLDeviceType`: Corresponding DLPack device type
"""
function gpu_backend_to_dldevice(backend::Symbol)
    backend_map = Dict(
        :CPU => LibTVMFFI.kDLCPU,
        :CUDA => LibTVMFFI.kDLCUDA,
        :ROCm => LibTVMFFI.kDLROCM,
        :AMDGPU => LibTVMFFI.kDLROCM,  # AMDGPU.jl uses ROCm
        :Metal => LibTVMFFI.kDLMetal,
        :oneAPI => LibTVMFFI.kDLExtDev,  # Intel oneAPI
        :Vulkan => LibTVMFFI.kDLVulkan,
        :OpenCL => LibTVMFFI.kDLOpenCL
    )

    if haskey(backend_map, backend)
        return backend_map[backend]
    else
        error("Unsupported GPU backend: $backend. Supported: $(keys(backend_map))")
    end
end

"""
    detect_backend(arr) -> (Symbol, Int)

Detect GPU backend and device ID from a GPU array.

Uses type-based detection and reflection to extract device ID.
Automatically handles wrapped arrays (OffsetArray, SubArray, etc.).

# Arguments
- `arr`: GPU array (CuArray, ROCArray, MtlArray, etc.), possibly wrapped

# Returns
- `(backend::Symbol, device_id::Int)`: Backend name and device ID

# Design Philosophy (Linus-style)
Good taste: Use type dispatch for backend-specific logic
Simple: Direct field access, no complex introspection
Practical: Fallback to device 0 if extraction fails
Robust: Unwrap array wrappers automatically

# Device ID Extraction Strategy
- CUDA.jl: Get device directly from array object
- AMDGPU.jl: Get device from array, convert from 1-indexed to 0-indexed
- Metal.jl: Always device 0 (single GPU)
- oneAPI.jl: Try device field, fallback to 0

# Examples
```julia
using CUDA
CUDA.device!(2)
x = CUDA.CuArray([1, 2, 3])
backend, dev_id = detect_backend(x)  # (:CUDA, 2)

# Also works with wrapped arrays!
using OffsetArrays
y = OffsetArray(x, -1:1)
backend, dev_id = detect_backend(y)  # (:CUDA, 2) - unwraps automatically
```
"""
function detect_backend(arr)
    # Unwrap any array wrappers to get to the actual GPU array
    unwrapped = _get_root_array(arr)

    arr_type = typeof(unwrapped)
    type_name = string(arr_type.name.name)

    # Detect backend from array type name and extract device ID
    if occursin("Cu", type_name) || occursin("CUDA", type_name)
        device_id = _extract_cuda_device_id(unwrapped)
        return (:CUDA, device_id)

    elseif occursin("ROC", type_name) || occursin("AMD", type_name)
        device_id = _extract_rocm_device_id(unwrapped)
        return (:ROCm, device_id)

    elseif occursin("Mtl", type_name) || occursin("Metal", type_name)
        # Metal.jl: Apple systems typically have single GPU
        return (:Metal, 0)

    elseif occursin("oneAPI", type_name)
        device_id = _extract_oneapi_device_id(unwrapped)
        return (:oneAPI, device_id)

    else
        # Assume CPU array
        return (:CPU, 0)
    end
end

"""
    _extract_cuda_device_id(arr) -> Int

Extract device ID from a CUDA array.

Design Philosophy (Linus-style):
- Good taste: Use parentmodule() to get the array's package directly
- No hacks: No Main.CUDA nonsense, no isdefined checks
- Direct: Call CUDA.device() and convert to Int

Strategy:
1. Get CUDA module via parentmodule(typeof(arr))
2. Call CUDA.device() to get current device (arrays are on current device)
3. Convert CuDevice to Int (0-indexed)
4. Fallback to 0 if anything fails

Note: CUDA.jl uses 0-indexed device IDs, matching DLPack standard.
"""
function _extract_cuda_device_id(arr)
    try
        # Navigate to root CUDA module
        cuda_module = _navigate_to_root_module(arr, :CUDA)

        # Get device ID
        return cuda_module.deviceid(cuda_module.device(arr))
    catch _
        # Silent fallback - could fail if CUDA API changed
        # Better to return 0 than crash user's code
    end

    return 0
end

"""
    _extract_rocm_device_id(arr) -> Int

Extract device ID from a ROCm/AMDGPU array.

Design Philosophy (Linus-style):
- Direct module access via parentmodule()
- Handle index conversion explicitly (AMDGPU 1-indexed → DLPack 0-indexed)
- Clear comments explaining the conversion

Strategy:
1. Get AMDGPU module via parentmodule()
2. Call AMDGPU.device_id() to get current device
3. Convert from 1-indexed (AMDGPU) to 0-indexed (DLPack)
4. Fallback to 0 if anything fails

Note: AMDGPU.jl uses 1-indexed device IDs, but DLPack uses 0-indexed.
"""
function _extract_rocm_device_id(arr)
    try
        # Navigate to AMDGPU root module
        amdgpu_module = _navigate_to_root_module(arr, :AMDGPU)

        # AMDGPU.device_id() returns 1-indexed (1, 2, 3, ...)
        dev_id_1indexed = amdgpu_module.device_id(amdgpu_module.device(arr))

        # DLPack uses 0-indexed device IDs
        # AMDGPU device 1 → DLPack device 0
        return dev_id_1indexed - 1
    catch _
        # Silent fallback
    end

    return 0
end

"""
    _extract_oneapi_device_id(arr) -> Int

Extract device ID from a oneAPI array.

Design Philosophy (Linus-style):
- Honest: We don't know oneAPI.jl internals yet
- Practical: Return 0 until someone with Intel hardware can test
- TODO: Needs investigation with actual oneAPI.jl code

Currently returns 0 (needs investigation of oneAPI.jl internals).
If you use oneAPI.jl and know how to get device ID, please contribute!
"""
function _extract_oneapi_device_id(arr)
    try
        # Navigate to oneAPI root module
        oneapi_module = _navigate_to_root_module(arr, :oneAPI)

        # TODO: Find the correct oneAPI.jl API for device queries
        # Possibilities:
        # - oneapi_module.device()
        # - oneapi_module.device_id()
        # - arr.queue.device or similar
        #
        # For now, assume device 0 (most common case)
    catch _
        # Silent fallback
    end

    return 0
end

# Deleted from_gpu_array - it's redundant!
# from_julia_array handles GPU arrays automatically.

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
# Note: _get_root_array and _navigate_to_root_module are now in utils.jl

# ============================================================================
# GPU Device ID Extraction
# ============================================================================

# Extend from_julia_array to handle GPU arrays
"""
    from_julia_array(arr::AbstractArray)

Extended method for GPU arrays - uses same DLTensorHolder as CPU!

# Design Philosophy (Linus-style)
ONE function, ONE type, ALL devices.
- Auto-detect GPU backend from array type
- Create appropriate device automatically
- Return unified DLTensorHolder

No from_gpu_array needed - it was a redundant special case!

# Examples
```julia
using CUDA

# CPU and GPU - same API!
cpu_holder = from_julia_array(cpu_arr)     # Auto: CPU device
gpu_holder = from_julia_array(gpu_arr)     # Auto: CUDA device

# Both return DLTensorHolder{T, S}
# Device info is in holder.tensor.device
```
"""
function from_julia_array(arr::S) where {S <: AbstractArray}
    T = eltype(arr)

    # Auto-detect backend and create GPU device
    backend, device_id = detect_backend(arr)
    dl_device_type = gpu_backend_to_dldevice(backend)
    device = DLDevice(Int32(dl_device_type), Int32(device_id))

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
    if backend == :Metal
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
    tensor = DLTensor(
        arr_ptr,
        device,
        Int32(length(shape_vec)),
        dt,
        pointer(shape_vec),
        pointer(strides_vec),
        byte_offset
    )

    return DLTensorHolder{T, S}(tensor, shape_vec, strides_vec, arr)
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
        backend, dev_id = detect_backend(arr)
        println("  Backend: $backend")
        println("  Device ID: $dev_id")
        println("  Element Type: ", eltype(arr))
        println("  Shape: ", size(arr))
        println("  Size: ", length(arr), " elements")
        println("  Memory Pointer: ", repr(UInt(pointer(arr))))

        # Map to DLDevice
        dl_type = gpu_backend_to_dldevice(backend)
        dl_dev = DLDevice(Int32(dl_type), Int32(dev_id))
        println("  DLDevice: ", dl_dev)

    catch e
        println("  Error getting info: ", e)
    end
end
