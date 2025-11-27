"""
Metal Extension for TVMFFI.jl

This extension enables zero-copy tensor exchange between Julia MtlArrays and TVM
on Apple Silicon.

# Usage
```julia
using TVMFFI
using Metal  # Triggers this extension

arr = Metal.MtlArray(Float32[1, 2, 3])
tensor = TVMTensor(arr)  # Zero-copy!
```
"""
module MetalExt

import TVMFFI
import TVMFFI: dldevice, _wrap_gpu_dltensor, _wrap_gpu_dltensor_view, DLDevice, LibTVMFFI,
               _register_wrapped_array, _is_contiguous, register_gpu_sync_callback,
               _ensure_cleanup_atexit_registered, TVMTensor
import Metal

# ============================================================================
# GPU Synchronization for cleanup
# ============================================================================

function __init__()
    # Register Metal sync callback for cleanup
    register_gpu_sync_callback() do
        if Metal.functional()
            Metal.synchronize()
        end
    end
    
    # Ensure cleanup atexit is registered
    _ensure_cleanup_atexit_registered()
end

# ============================================================================
# Device Detection
# ============================================================================

function TVMFFI.dldevice(x::Metal.MtlArray)
    return DLDevice(Int32(LibTVMFFI.kDLMetal), Int32(0))
end

function TVMFFI.dldevice(x::SubArray{T, N, <:Metal.MtlArray}) where {T, N}
    return TVMFFI.dldevice(parent(x))
end

# ============================================================================
# Metal Tensor Wrapping (TVM → Julia) - SIMPLIFIED
# ============================================================================

const _METAL = Int32(LibTVMFFI.kDLMetal)

function TVMFFI._wrap_gpu_dltensor(::Val{_METAL}, ::Type{T}, data_ptr::Ptr{Cvoid},
        shape::Vector{Int64}, strides::Vector{Int64},
        owner::TVMTensor) where {T}
    if _is_contiguous(shape, strides)
        mtl_ptr = Metal.MtlPtr{T}(UInt(data_ptr))
        arr = unsafe_wrap(Metal.MtlArray, mtl_ptr, Tuple(shape))
        _register_wrapped_array(arr, owner)
        return arr
    else
        error("Non-contiguous GPU arrays (strides=$strides) are not supported. " *
              "Please use `collect(slice)` to create a contiguous copy before passing to TVM.")
    end
end

# ============================================================================
# GPU Tensor View (for callbacks, no owner)
# ============================================================================

function TVMFFI._wrap_gpu_dltensor_view(::Val{_METAL}, ::Type{T}, data_ptr::Ptr{Cvoid},
        shape::NTuple{N, Int64}, strides::NTuple{N, Int64}) where {T, N}
    if N == 0 || _is_tuple_contiguous(shape, strides)
        mtl_ptr = Metal.MtlPtr{T}(UInt(data_ptr))
        return unsafe_wrap(Metal.MtlArray, mtl_ptr, shape)
    else
        error("Non-contiguous GPU arrays (strides=$strides) are not supported in callbacks.")
    end
end

function _is_tuple_contiguous(shape::NTuple{N, Int64}, strides::NTuple{N, Int64}) where {N}
    N == 0 && return true
    expected = Int64(1)
    for i in 1:N
        strides[i] != expected && return false
        expected *= shape[i]
    end
    return true
end

end # module MetalExt
