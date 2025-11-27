"""
AMDGPU Extension for TVMFFI.jl

This extension enables zero-copy tensor exchange between Julia ROCArrays and TVM
on AMD GPUs.

# Usage
```julia
using TVMFFI
using AMDGPU  # Triggers this extension

arr = AMDGPU.ROCArray(Float32[1, 2, 3])
tensor = TVMTensor(arr)  # Zero-copy!
```
"""
module AMDGPUExt

import TVMFFI
import TVMFFI: dldevice, _wrap_gpu_dltensor, _wrap_gpu_dltensor_view, DLDevice, LibTVMFFI,
               _register_wrapped_array, _is_contiguous, register_gpu_sync_callback,
               _ensure_cleanup_atexit_registered, TVMTensor
import AMDGPU

# ============================================================================
# GPU Synchronization for cleanup
# ============================================================================

function __init__()
    # Register AMDGPU sync callback for cleanup
    register_gpu_sync_callback() do
        if AMDGPU.functional()
            AMDGPU.synchronize()
        end
    end
    
    # Ensure cleanup atexit is registered
    _ensure_cleanup_atexit_registered()
end

# ============================================================================
# Device Detection
# ============================================================================

function TVMFFI.dldevice(x::AMDGPU.ROCArray)
    dev_id_1indexed = AMDGPU.device_id(AMDGPU.device(x))
    dev_id_0indexed = dev_id_1indexed - 1
    return DLDevice(Int32(LibTVMFFI.kDLROCM), Int32(dev_id_0indexed))
end

function TVMFFI.dldevice(x::SubArray{T, N, <:AMDGPU.ROCArray}) where {T, N}
    return TVMFFI.dldevice(parent(x))
end

# ============================================================================
# ROCm Tensor Wrapping (TVM → Julia) - SIMPLIFIED
# ============================================================================

const _ROCM = Int32(LibTVMFFI.kDLROCM)
const _ROCM_HOST = Int32(LibTVMFFI.kDLROCMHost)

function TVMFFI._wrap_gpu_dltensor(::Val{_ROCM}, ::Type{T}, data_ptr::Ptr{Cvoid},
        shape::Vector{Int64}, strides::Vector{Int64},
        owner::TVMTensor) where {T}
    _wrap_rocm_dltensor(T, data_ptr, shape, strides, owner)
end

function TVMFFI._wrap_gpu_dltensor(::Val{_ROCM_HOST}, ::Type{T}, data_ptr::Ptr{Cvoid},
        shape::Vector{Int64}, strides::Vector{Int64},
        owner::TVMTensor) where {T}
    _wrap_rocm_dltensor(T, data_ptr, shape, strides, owner)
end

function _wrap_rocm_dltensor(::Type{T}, data_ptr::Ptr{Cvoid}, shape::Vector{Int64},
        strides::Vector{Int64}, owner::TVMTensor) where {T}
    if _is_contiguous(shape, strides)
        roc_ptr = Ptr{T}(UInt(data_ptr))
        arr = unsafe_wrap(AMDGPU.ROCArray, roc_ptr, Tuple(shape))
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

function TVMFFI._wrap_gpu_dltensor_view(::Val{_ROCM}, ::Type{T}, data_ptr::Ptr{Cvoid},
        shape::NTuple{N, Int64}, strides::NTuple{N, Int64}) where {T, N}
    _wrap_rocm_view(T, data_ptr, shape, strides)
end

function TVMFFI._wrap_gpu_dltensor_view(::Val{_ROCM_HOST}, ::Type{T}, data_ptr::Ptr{Cvoid},
        shape::NTuple{N, Int64}, strides::NTuple{N, Int64}) where {T, N}
    _wrap_rocm_view(T, data_ptr, shape, strides)
end

function _wrap_rocm_view(::Type{T}, data_ptr::Ptr{Cvoid},
        shape::NTuple{N, Int64}, strides::NTuple{N, Int64}) where {T, N}
    if N == 0 || _is_tuple_contiguous(shape, strides)
        roc_ptr = Ptr{T}(UInt(data_ptr))
        return unsafe_wrap(AMDGPU.ROCArray, roc_ptr, shape)
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

end # module AMDGPUExt
