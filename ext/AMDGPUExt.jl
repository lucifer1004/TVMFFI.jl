"""
AMDGPU Extension for TVMFFI.jl

This extension is automatically loaded when AMDGPU.jl is imported.
It enables zero-copy tensor exchange between Julia ROCArrays and TVM on AMD GPUs.

# Usage
```julia
using TVMFFI
using AMDGPU  # Triggers this extension

arr = AMDGPU.ROCArray(Float32[1, 2, 3])
tensor = TVMTensor(arr)  # Zero-copy!
```

# Design Philosophy (Linus-style)
- Use type dispatch instead of string matching
- Direct API calls, no _navigate_to_root_module hacks
- AMDGPU.jl knows AMDGPU.jl best
"""
module AMDGPUExt

import TVMFFI
import AMDGPU
import DLPack

# ============================================================================
# DLPack Integration (zero-copy tensor exchange)
# ============================================================================
# Note: Backend detection is handled by DLPack.dldevice - no duplication!

# Extend DLPack.share for ROCArrays
function DLPack.share(A::AMDGPU.ROCArray)
    DLPack.unsafe_share(A)
end

# Handle strided views
function DLPack.share(A::SubArray{T, N, <:AMDGPU.ROCArray}) where {T, N}
    DLPack.unsafe_share(A)
end

# Extend DLPack.dldevice for ROCArrays
function DLPack.dldevice(x::AMDGPU.ROCArray)
    # AMDGPU.jl uses 1-indexed device IDs, DLPack uses 0-indexed
    dev_id_1indexed = AMDGPU.device_id(AMDGPU.device(x))
    dev_id_0indexed = dev_id_1indexed - 1
    return DLPack.DLDevice(DLPack.kDLROCM, Cint(dev_id_0indexed))
end

function DLPack.dldevice(x::SubArray{T, N, <:AMDGPU.ROCArray}) where {T, N}
    return DLPack.dldevice(parent(x))
end

# Extend jlarray_type for wrapping DLPack tensors back to ROCArray
DLPack.jlarray_type(::Val{DLPack.kDLROCM}) = AMDGPU.ROCArray
DLPack.jlarray_type(::Val{DLPack.kDLROCMHost}) = AMDGPU.ROCArray

# Extend unsafe_wrap for ROCArray
function Base.unsafe_wrap(::Type{<:AMDGPU.ROCArray}, manager::DLPack.DLManager{T}) where {T}
    if DLPack.device_type(manager) in (DLPack.kDLROCM, DLPack.kDLROCMHost)
        addr = DLPack.pointer(manager)
        sz = DLPack.unsafe_size(manager)
        # AMDGPU uses ROCDeviceArray or similar pointer type
        return unsafe_wrap(AMDGPU.ROCArray, Ptr{T}(addr), sz)
    end
    throw(ArgumentError("Only ROCm arrays can be wrapped with ROCArray"))
end

end # module AMDGPUExt

