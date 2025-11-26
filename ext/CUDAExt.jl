"""
CUDA Extension for TVMFFI.jl

This extension is a placeholder - the actual DLPack integration for CUDA
is provided by DLPack.jl's own CUDAExt.

When user does `using TVMFFI; using CUDA`:
- DLPack.jl's CUDAExt handles: DLPack.share, DLPack.dldevice, jlarray_type, unsafe_wrap
- TVMFFI uses these through the DLPack interface

# Design Philosophy (Linus-style)
- Don't duplicate what DLPack.jl already provides
- Empty module is better than redundant code
- Keep the extension file for potential future TVMFFI-specific CUDA functionality
"""
module CUDAExt

# DLPack.jl's CUDAExt already provides:
# - DLPack.share(::CUDA.StridedCuArray)
# - DLPack.dldevice(::CUDA.StridedCuArray)
# - DLPack.jlarray_type(::Val{kDLCUDA})
# - Base.unsafe_wrap(::Type{<:CUDA.CuArray}, ::DLPack.DLManager)
#
# No need to duplicate here!

end # module CUDAExt
