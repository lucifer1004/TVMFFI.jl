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
    dlpack.jl

DLPack zero-copy tensor exchange for TVMFFI.

This module provides zero-copy conversion between Julia arrays and TVM tensors
using the DLPack protocol. It is self-contained and does NOT depend on DLPack.jl.

# Key Functions
- `TVMTensor(arr)`: Convert Julia array to TVMTensor (zero-copy)
- `from_dlpack(tensor)`: Convert TVMTensor to Julia array (zero-copy)
- `to_dlmanaged_tensor(view)`: Export TensorView as DLManagedTensor

# Design
- Uses TensorView as the unified internal representation
- DLManagedTensor is only used for external exchange
- Proper lifecycle management via ownership enum
"""

using .LibTVMFFI

#=============================================================================
  Julia Array → TVMTensor (zero-copy)
=============================================================================#

"""
    TVMTensor(arr::StridedArray) -> TVMTensor

Convert a Julia array to a TVMTensor via DLPack protocol (zero-copy).

The returned TVMTensor holds a reference to the Julia array through
TVM's reference counting mechanism. The array will be kept alive
as long as the TVMTensor (or any copies of it) exists.

# Example
```julia
arr = rand(Float32, 3, 4)
tensor = TVMTensor(arr)
# tensor shares memory with arr
# arr is kept alive by TVM's reference counting
```

# Note
This creates a `kTVMFFITensor` object (type_index=70) which has proper
lifecycle management via reference counting, unlike `TensorView` which
creates a raw `kTVMFFIDLTensorPtr` (type_index=7).

# See also
- [`from_dlpack`](@ref): Convert TVMTensor back to Julia array
- [`TensorView`](@ref): Lightweight view without reference counting
"""
function TVMTensor(arr::StridedArray{T, N}) where {T, N}
    # Step 1: Create TensorView (handles shape, strides, device detection)
    view = TensorView(arr)

    # Step 2: Export as DLManagedTensor
    dlmanaged_ptr = to_dlmanaged_tensor(view)

    # Step 3: Call TVM C API to create TVMTensor
    ret,
    tensor_handle = LibTVMFFI.TVMFFITensorFromDLPack(
        Ptr{Cvoid}(dlmanaged_ptr),
        Int32(0),  # no alignment requirement
        Int32(0)   # no contiguity requirement
    )

    if ret != 0
        # Clean up on failure - remove from pool
        lock(_DLMANAGED_POOL_LOCK) do
            delete!(_DLMANAGED_POOL, Ptr{Cvoid}(dlmanaged_ptr))
        end
        error("Failed to create TVMTensor from DLPack (ret=$ret)")
    end

    # Note: TVM now owns the DLManagedTensor and will call our deleter
    # when it's done. The deleter will release the array from the pool.

    return TVMTensor(tensor_handle; borrowed = false)
end

#=============================================================================
  TVMTensor → Julia Array (zero-copy)
=============================================================================#

"""
    from_dlpack(tensor::TVMTensor) -> AbstractArray

Convert TVMTensor to Julia array (zero-copy).

The returned array shares memory with the TVMTensor. The TVMTensor
is kept alive automatically until the array is garbage collected.

# Example
```julia
tensor = some_tvm_function()
arr = from_dlpack(tensor)  # Zero-copy, shares memory
```

# See also
- [`TVMTensor`](@ref): Create TVMTensor from Julia array
"""
function from_dlpack(tensor::TVMTensor)
    # Step 1: Get DLTensor pointer directly from TVMTensor
    # No allocation, no new refcounting path
    dltensor_ptr = get_dltensor_ptr(tensor)
    dltensor = unsafe_load(dltensor_ptr)

    # Step 2: Extract type and shape
    T = dtype_to_julia_type(dltensor.dtype)
    ndim = Int(dltensor.ndim)

    shape_vec = if ndim > 0
        unsafe_wrap(Array, dltensor.shape, ndim) |> copy
    else
        Int64[]
    end

    strides_vec = if dltensor.strides != C_NULL && ndim > 0
        unsafe_wrap(Array, dltensor.strides, ndim) |> copy
    else
        _compute_contiguous_strides(shape_vec)
    end

    # Step 3: Wrap as Julia array based on device
    device_type = dltensor.device.device_type

    if device_type == Int32(LibTVMFFI.kDLCPU)
        return _wrap_cpu_dltensor(T, dltensor.data, shape_vec, strides_vec, tensor)
    else
        return _wrap_gpu_dltensor(
            Val(device_type), T, dltensor.data, shape_vec, strides_vec, tensor)
    end
end

"""
Wrap a CPU DLTensor data pointer as Julia Array.
"""
function _wrap_cpu_dltensor(::Type{T}, data_ptr::Ptr{Cvoid}, shape::Vector{Int64},
        strides::Vector{Int64}, owner::TVMTensor) where {T}
    # Check contiguity
    if _is_contiguous(shape, strides)
        # Zero-copy wrap
        arr = unsafe_wrap(Array, Ptr{T}(data_ptr), Tuple(shape))
        _register_wrapped_array(arr, owner)
        return arr
    else
        # Non-contiguous: must copy
        @warn "Non-contiguous tensor detected, copying data"
        return _copy_strided_data(T, data_ptr, shape, strides)
    end
end

"""
Wrap a GPU DLTensor data pointer as GPU Array.
Dispatches to appropriate GPU backend via Val{device_type}.
"""
function _wrap_gpu_dltensor(::Val{D}, ::Type{T}, data_ptr::Ptr{Cvoid},
        shape::Vector{Int64}, strides::Vector{Int64},
        owner::TVMTensor) where {D, T}
    device_name = _device_type_to_name(Int32(D))
    error("GPU tensor on $device_name device requires the appropriate package. " *
          "Please load CUDA.jl, Metal.jl, or AMDGPU.jl as needed.")
end

"""
Wrap a GPU DLTensor as a temporary view (no owner).

This is used in callback contexts where the caller guarantees the data
remains valid during the callback. The returned array should not be used
after the callback returns.

Extensions can override this for specific GPU backends.
"""
function _wrap_gpu_dltensor_view(::Val{D}, ::Type{T}, data_ptr::Ptr{Cvoid},
        shape::NTuple{N, Int64}, strides::NTuple{N, Int64}) where {D, T, N}
    device_name = _device_type_to_name(Int32(D))
    error("GPU tensor view on $device_name device requires the appropriate package. " *
          "Please load CUDA.jl, Metal.jl, or AMDGPU.jl as needed.")
end

"""
Check if strides indicate contiguous memory layout.
"""
function _is_contiguous(shape::Vector{Int64}, strides::Vector{Int64})
    ndim = length(shape)
    ndim == 0 && return true

    # Check column-major (Fortran) contiguity for Julia
    expected_stride = Int64(1)
    for i in 1:ndim
        if strides[i] != expected_stride
            return false
        end
        expected_stride *= shape[i]
    end
    return true
end

"""
Copy strided data to contiguous array.
"""
function _copy_strided_data(::Type{T}, data_ptr::Ptr{Cvoid}, shape::Vector{Int64},
        strides::Vector{Int64}) where {T}
    result = Array{T}(undef, Tuple(shape)...)
    _copy_strided!(result, Ptr{T}(data_ptr), shape, strides)
    return result
end

#=============================================================================
  Lifecycle Management Pool
=============================================================================#

# Pool to keep TVMTensor owners alive while wrapped arrays exist
# Key: objectid(arr), Value: TVMTensor owner
# 
# Simple design: When array is GC'd, owner is released, owner's finalizer
# calls DecRef, TVM cleans up. No complex deleter management needed.
const _WRAPPED_ARRAYS = Dict{UInt, TVMTensor}()
const _WRAPPED_ARRAYS_LOCK = ReentrantLock()
const _WRAPPED_ARRAYS_ATEXIT_REGISTERED = Ref(false)

# GPU synchronization callbacks - extensions register their sync functions
const _GPU_SYNC_CALLBACKS = Function[]
const _GPU_SYNC_LOCK = ReentrantLock()

"""
Register a GPU synchronization callback for cleanup.
Extensions call this to ensure GPU operations complete before cleanup.
"""
function register_gpu_sync_callback(callback::Function)
    lock(_GPU_SYNC_LOCK) do
        push!(_GPU_SYNC_CALLBACKS, callback)
    end
end

"""
Cleanup wrapped arrays at exit.

Strategy: Clear TVMTensor handles to prevent finalizers from calling DecRef
during Julia's shutdown when TVM library may be in an invalid state.
"""
function _cleanup_wrapped_arrays_at_exit()
    _julia_is_exiting[] = true

    # Synchronize GPU operations first
    lock(_GPU_SYNC_LOCK) do
        for callback in _GPU_SYNC_CALLBACKS
            try
                callback()
            catch
                # Ignore errors during cleanup
            end
        end
    end

    # Force GC to run TVM finalizers while GPU context is still valid
    # This is critical for GPU arrays that may have TVM references
    GC.gc()

    # Sync again after GC (finalizers may have queued GPU work)
    lock(_GPU_SYNC_LOCK) do
        for callback in _GPU_SYNC_CALLBACKS
            try
                callback()
            catch
            end
        end
    end

    lock(_WRAPPED_ARRAYS_LOCK) do
        for (_, owner) in _WRAPPED_ARRAYS
            owner.handle = C_NULL  # Prevent finalizer DecRef
        end
        empty!(_WRAPPED_ARRAYS)
    end
end

"""
    _ensure_cleanup_atexit_registered()

Ensure the cleanup atexit handler is registered.
Called by GPU extensions to ensure proper cleanup order.
"""
function _ensure_cleanup_atexit_registered()
    if !_WRAPPED_ARRAYS_ATEXIT_REGISTERED[]
        atexit(_cleanup_wrapped_arrays_at_exit)
        _WRAPPED_ARRAYS_ATEXIT_REGISTERED[] = true
    end
end

"""
    _register_wrapped_array(arr, owner)

Keep TVMTensor owner alive while wrapped array exists.

When array is GC'd:
1. Remove from _WRAPPED_ARRAYS
2. TVMTensor becomes unreachable → GC'd → finalizer calls DecRef
3. TVM cleans up

Simple design: just reference holding, no complex deleter management.
"""
function _register_wrapped_array(arr, owner::TVMTensor)
    _ensure_cleanup_atexit_registered()

    lock(_WRAPPED_ARRAYS_LOCK) do
        _WRAPPED_ARRAYS[objectid(arr)] = owner
    end

    finalizer(arr) do a
        _julia_is_exiting[] && return
        lock(_WRAPPED_ARRAYS_LOCK) do
            delete!(_WRAPPED_ARRAYS, objectid(a))
        end
    end
end

"""
Unregister a wrapped array from lifecycle management.
Used when identity optimization transfers ownership back to caller.
"""
function _unregister_wrapped_array(arr)
    lock(_WRAPPED_ARRAYS_LOCK) do
        if haskey(_WRAPPED_ARRAYS, objectid(arr))
            owner = _WRAPPED_ARRAYS[objectid(arr)]
            owner.handle = C_NULL  # Prevent finalizer DecRef
            delete!(_WRAPPED_ARRAYS, objectid(arr))
        end
    end
end

"""
Copy data from strided memory to contiguous array.
"""
function _copy_strided!(dst::Array{T}, src_ptr::Ptr{T}, shape::Vector{Int64},
        strides::Vector{Int64}) where {T}
    ndim = length(shape)

    if ndim == 1
        # 1D case: simple strided copy
        for i in 1:shape[1]
            dst[i] = unsafe_load(src_ptr, 1 + (i - 1) * strides[1])
        end
    elseif ndim == 2
        # 2D case: optimized
        for j in 1:shape[2]
            for i in 1:shape[1]
                offset = 1 + (i - 1) * strides[1] + (j - 1) * strides[2]
                dst[i, j] = unsafe_load(src_ptr, offset)
            end
        end
    else
        # General case: recursive
        _copy_strided_recursive!(dst, src_ptr, shape, strides, 1, 0)
    end
end

function _copy_strided_recursive!(dst::Array{T}, src_ptr::Ptr{T},
        shape::Vector{Int64}, strides::Vector{Int64},
        dim::Int, src_offset::Int) where {T}
    if dim == length(shape)
        # Last dimension: copy elements
        for i in 1:shape[dim]
            dst_idx = LinearIndices(dst)[ntuple(d -> d == dim ? i : 1, length(shape))...]
            dst[dst_idx] = unsafe_load(src_ptr, 1 + src_offset + (i - 1) * strides[dim])
        end
    else
        # Recurse to next dimension
        for i in 1:shape[dim]
            new_offset = src_offset + (i - 1) * strides[dim]
            _copy_strided_recursive!(dst, src_ptr, shape, strides, dim + 1, new_offset)
        end
    end
end

#=============================================================================
  GPU Tensor Wrapping (extensible via Julia's multiple dispatch)
=============================================================================#

# Default fallback - GPU extensions add methods for specific device types

function _device_type_to_name(device_type::Int32)
    names = Dict(
        Int32(LibTVMFFI.kDLCUDA) => "CUDA",
        Int32(LibTVMFFI.kDLCUDAHost) => "CUDA Host",
        Int32(LibTVMFFI.kDLCUDAManaged) => "CUDA Managed",
        Int32(LibTVMFFI.kDLROCM) => "ROCm",
        Int32(LibTVMFFI.kDLMetal) => "Metal",
        Int32(LibTVMFFI.kDLVulkan) => "Vulkan",
        Int32(LibTVMFFI.kDLOpenCL) => "OpenCL",
        Int32(LibTVMFFI.kDLOneAPI) => "oneAPI"
    )
    return get(names, device_type, "Unknown (type=$device_type)")
end

#=============================================================================
  DLMANAGED_POOL Cleanup at Exit
=============================================================================#

const _DLMANAGED_POOL_ATEXIT_REGISTERED = Ref(false)

"""
Cleanup _DLMANAGED_POOL at exit.

This is CRITICAL for GPU arrays: the pool holds references to DLManagedTensor
structures that point to GPU memory. We must release these BEFORE TVM/CUDA
libraries are unloaded, otherwise their cleanup routines will crash.

Strategy: Clear the pool entries to allow their underlying data to be released.
The DLManagedTensor deleters won't be called (we just drop references), but
this is acceptable - the GPU memory will be cleaned up when CUDA context is
destroyed anyway.
"""
function _cleanup_dlmanaged_pool_at_exit()
    _julia_is_exiting[] = true

    lock(_DLMANAGED_POOL_LOCK) do
        empty!(_DLMANAGED_POOL)
    end
end

#=============================================================================
  Module Initialization
=============================================================================#

function _init_dlpack_api()
    # Initialize the DLManagedTensor deleter function pointer
    _init_dlmanaged_deleter()

    # Register atexit handler for _DLMANAGED_POOL cleanup
    # This must run BEFORE _cleanup_wrapped_arrays_at_exit
    if !_DLMANAGED_POOL_ATEXIT_REGISTERED[]
        atexit(_cleanup_dlmanaged_pool_at_exit)
        _DLMANAGED_POOL_ATEXIT_REGISTERED[] = true
    end
end
