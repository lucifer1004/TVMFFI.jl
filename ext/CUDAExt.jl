"""
CUDA Extension for TVMFFI.jl

This extension enables zero-copy tensor exchange between Julia CuArrays and TVM.

# Usage
```julia
using TVMFFI
using CUDA  # Triggers this extension

arr = CUDA.CuArray(Float32[1, 2, 3])
tensor = TVMTensor(arr)  # Zero-copy!
```
"""
module CUDAExt

import TVMFFI
import TVMFFI: dldevice, _wrap_gpu_dltensor, _wrap_gpu_dltensor_view, DLDevice, LibTVMFFI,
               _register_wrapped_array, _is_contiguous, register_gpu_sync_callback,
               _ensure_cleanup_atexit_registered, TVMTensor
import CUDA

# ============================================================================
# GPU Synchronization for cleanup
# ============================================================================

function __init__()
    # Register CUDA sync callback for cleanup
    # This is called by _cleanup_wrapped_arrays_at_exit before GC.gc()
    register_gpu_sync_callback() do
        if CUDA.functional()
            CUDA.synchronize()
        end
    end

    # Ensure cleanup atexit is registered
    # This guarantees proper cleanup order: sync → GC → sync → clear handles
    _ensure_cleanup_atexit_registered()
end

# ============================================================================
# Device Detection
# ============================================================================

function TVMFFI.dldevice(x::CUDA.CuArray)
    dev = CUDA.device(x)
    dev_id = dev.handle  # 0-indexed
    return DLDevice(Int32(LibTVMFFI.kDLCUDA), Int32(dev_id))
end

function TVMFFI.dldevice(x::SubArray{T, N, <:CUDA.CuArray}) where {T, N}
    return TVMFFI.dldevice(parent(x))
end

# ============================================================================
# CUDA Tensor Wrapping (TVM → Julia) - SIMPLIFIED
# ============================================================================

# Device type constants
const _CUDA = Int32(LibTVMFFI.kDLCUDA)
const _CUDA_HOST = Int32(LibTVMFFI.kDLCUDAHost)
const _CUDA_MANAGED = Int32(LibTVMFFI.kDLCUDAManaged)

# Main CUDA device
function TVMFFI._wrap_gpu_dltensor(::Val{_CUDA}, ::Type{T}, data_ptr::Ptr{Cvoid},
        shape::Vector{Int64}, strides::Vector{Int64},
        owner::TVMTensor) where {T}
    _wrap_cuda_dltensor(T, data_ptr, shape, strides, owner)
end

# CUDA Host memory
function TVMFFI._wrap_gpu_dltensor(::Val{_CUDA_HOST}, ::Type{T}, data_ptr::Ptr{Cvoid},
        shape::Vector{Int64}, strides::Vector{Int64},
        owner::TVMTensor) where {T}
    _wrap_cuda_dltensor(T, data_ptr, shape, strides, owner)
end

# CUDA Managed memory
function TVMFFI._wrap_gpu_dltensor(::Val{_CUDA_MANAGED}, ::Type{T}, data_ptr::Ptr{Cvoid},
        shape::Vector{Int64}, strides::Vector{Int64},
        owner::TVMTensor) where {T}
    _wrap_cuda_dltensor(T, data_ptr, shape, strides, owner)
end

function _wrap_cuda_dltensor(::Type{T}, data_ptr::Ptr{Cvoid}, shape::Vector{Int64},
        strides::Vector{Int64}, owner::TVMTensor) where {T}
    if _is_contiguous(shape, strides)
        # Zero-copy: wrap as CuArray
        cu_ptr = CUDA.CuPtr{T}(UInt(data_ptr))
        arr = unsafe_wrap(CUDA.CuArray, cu_ptr, Tuple(shape))

        # Simple lifecycle: just keep owner alive
        _register_wrapped_array(arr, owner)

        return arr
    else
        # Non-contiguous GPU arrays are not supported
        # unsafe_wrap assumes contiguous memory, and GPU strided copy requires kernel launch
        error("Non-contiguous GPU arrays (strides=$strides) are not supported. " *
              "Please use `collect(slice)` to create a contiguous copy before passing to TVM.")
    end
end

# ============================================================================
# GPU Tensor View (for callbacks, no owner)
# ============================================================================

"""
Wrap CUDA DLTensor as temporary CuArray view (no lifecycle management).

Used in callback contexts where the caller guarantees data validity.
The returned CuArray should not be stored or used after the callback returns.
"""
function TVMFFI._wrap_gpu_dltensor_view(::Val{_CUDA}, ::Type{T}, data_ptr::Ptr{Cvoid},
        shape::NTuple{N, Int64}, strides::NTuple{N, Int64}) where {T, N}
    _wrap_cuda_view(T, data_ptr, shape, strides)
end

function TVMFFI._wrap_gpu_dltensor_view(::Val{_CUDA_HOST}, ::Type{T}, data_ptr::Ptr{Cvoid},
        shape::NTuple{N, Int64}, strides::NTuple{N, Int64}) where {T, N}
    _wrap_cuda_view(T, data_ptr, shape, strides)
end

function TVMFFI._wrap_gpu_dltensor_view(
        ::Val{_CUDA_MANAGED}, ::Type{T}, data_ptr::Ptr{Cvoid},
        shape::NTuple{N, Int64}, strides::NTuple{N, Int64}) where {T, N}
    _wrap_cuda_view(T, data_ptr, shape, strides)
end

function _wrap_cuda_view(::Type{T}, data_ptr::Ptr{Cvoid},
        shape::NTuple{N, Int64}, strides::NTuple{N, Int64}) where {T, N}
    # Check contiguity
    if N == 0 || _is_tuple_contiguous(shape, strides)
        cu_ptr = CUDA.CuPtr{T}(UInt(data_ptr))
        return unsafe_wrap(CUDA.CuArray, cu_ptr, shape)
    else
        error("Non-contiguous GPU arrays (strides=$strides) are not supported in callbacks.")
    end
end

# Check contiguity for tuple strides
function _is_tuple_contiguous(shape::NTuple{N, Int64}, strides::NTuple{N, Int64}) where {N}
    N == 0 && return true
    expected = Int64(1)
    for i in 1:N
        strides[i] != expected && return false
        expected *= shape[i]
    end
    return true
end

end # module CUDAExt
