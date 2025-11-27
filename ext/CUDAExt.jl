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
import TVMFFI: dldevice, _wrap_gpu_dltensor, DLDevice, LibTVMFFI,
               _register_wrapped_array, _is_contiguous, register_gpu_sync_callback,
               TVMTensor
import CUDA

# ============================================================================
# GPU Synchronization for cleanup
# ============================================================================

function __init__()
    # Register CUDA sync callback for cleanup
    register_gpu_sync_callback() do
        if CUDA.functional()
            CUDA.synchronize()
        end
    end
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

end # module CUDAExt
