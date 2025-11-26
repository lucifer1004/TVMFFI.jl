"""
Metal Extension for TVMFFI.jl

This extension is automatically loaded when Metal.jl is imported.
It enables zero-copy tensor exchange between Julia MtlArrays and TVM on Apple Silicon.

# Usage
```julia
using TVMFFI
using Metal  # Triggers this extension

arr = Metal.MtlArray(Float32[1, 2, 3])
tensor = TVMTensor(arr)  # Zero-copy!
```

# Design Philosophy (Linus-style)
- Use type dispatch instead of string matching
- Direct API calls, no _navigate_to_root_module hacks
- Metal.jl knows Metal.jl best
"""
module MetalExt

import TVMFFI
import Metal
import DLPack

# ============================================================================
# DLPack Integration (zero-copy tensor exchange)
# ============================================================================
# Note: Backend detection is handled by DLPack.dldevice - no duplication!

# Extend DLPack.share for MtlArrays
function DLPack.share(A::Metal.MtlArray)
    DLPack.unsafe_share(A)
end

# Handle strided views
function DLPack.share(A::SubArray{T, N, <:Metal.MtlArray}) where {T, N}
    DLPack.unsafe_share(A)
end

# Extend DLPack.dldevice for MtlArrays
# Metal on Apple Silicon typically has single GPU (device 0)
function DLPack.dldevice(x::Metal.MtlArray)
    return DLPack.DLDevice(DLPack.kDLMetal, Cint(0))
end

function DLPack.dldevice(x::SubArray{T, N, <:Metal.MtlArray}) where {T, N}
    return DLPack.DLDevice(DLPack.kDLMetal, Cint(0))
end

# Extend jlarray_type for wrapping DLPack tensors back to MtlArray
DLPack.jlarray_type(::Val{DLPack.kDLMetal}) = Metal.MtlArray

# Extend unsafe_wrap for MtlArray
function Base.unsafe_wrap(::Type{<:Metal.MtlArray}, manager::DLPack.DLManager{T}) where {T}
    if DLPack.device_type(manager) == DLPack.kDLMetal
        addr = DLPack.pointer(manager)
        sz = DLPack.unsafe_size(manager)
        # Metal uses MtlPtr for GPU pointers
        return unsafe_wrap(Metal.MtlArray, Metal.MtlPtr{T}(addr), sz)
    end
    throw(ArgumentError("Only Metal arrays can be wrapped with MtlArray"))
end

end # module MetalExt

