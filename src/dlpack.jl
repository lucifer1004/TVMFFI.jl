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
using the DLPack protocol. It leverages DLPack.jl for the underlying DLPack
infrastructure.

# Key Functions
- `TVMTensor(arr)`: Convert Julia array to TVMTensor (zero-copy)
- `from_dlpack(tensor)`: Convert TVMTensor to Julia array (zero-copy)

# Design
Instead of using `kTVMFFIDLTensorPtr` (raw pointer, no lifecycle management),
we use `kTVMFFITensor` (reference-counted object) for safe zero-copy exchange.
"""

using .LibTVMFFI
import DLPack
import DLPack: from_dlpack

#=============================================================================
  Julia Array → TVMTensor (zero-copy)
=============================================================================#

# Pool to keep Julia arrays alive while TVM holds references
# Key: pointer to DLManagedTensor, Value: (capsule, array)
const _DLPACK_SHARES_POOL = Dict{Ptr{Cvoid}, Any}()
const _DLPACK_POOL_LOCK = ReentrantLock()

"""
    _dlpack_deleter(ptr::Ptr{Cvoid})

Callback function called by TVM when it no longer needs the tensor.
This releases the Julia array from the shares pool.
"""
function _dlpack_deleter(manager_ctx::Ptr{Cvoid})
    lock(_DLPACK_POOL_LOCK) do
        delete!(_DLPACK_SHARES_POOL, manager_ctx)
    end
    return nothing
end

# C function pointer for deleter - initialized in __init__
const _DLPACK_DELETER_CPTR = Ref{Ptr{Cvoid}}(C_NULL)

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
    # Step 1: Create DLPack capsule using DLPack.jl
    capsule = DLPack.share(arr)
    
    # Step 2: Get the DLManagedTensor from capsule
    # DLPack.jl uses DLManagedTensor (v0.x), not DLManagedTensorVersioned
    managed = capsule.tensor
    
    # Step 3: Set up the deleter to release the array when TVM is done
    # We need to modify the DLManagedTensor to use our deleter
    managed_ptr = pointer_from_objref(managed)
    
    # Store in pool before calling C API
    lock(_DLPACK_POOL_LOCK) do
        _DLPACK_SHARES_POOL[managed_ptr] = (capsule, arr)
    end
    
    # Update the DLManagedTensor's deleter and manager_ctx
    # Since DLManagedTensor is mutable, we can modify it
    managed.manager_ctx = managed_ptr
    managed.deleter = _DLPACK_DELETER_CPTR[]
    
    # Step 4: Call TVM C API to create TVMTensor
    # Use the v0.x API since DLPack.jl provides DLManagedTensor
    ret, tensor_handle = LibTVMFFI.TVMFFITensorFromDLPack(
        managed_ptr,
        Int32(0),  # no alignment requirement
        Int32(0)   # no contiguity requirement (we handle strides)
    )
    
    if ret != 0
        # Clean up on failure
        lock(_DLPACK_POOL_LOCK) do
            delete!(_DLPACK_SHARES_POOL, managed_ptr)
        end
        error("Failed to create TVMTensor from DLPack (ret=$ret)")
    end
    
    # Note: TVM now owns the DLManagedTensor and will call our deleter
    # when it's done. The deleter will release the array from the pool.
    
    return TVMTensor(tensor_handle; borrowed=false)
end

#=============================================================================
  TVMTensor → Julia Array (zero-copy)
=============================================================================#

"""
    from_dlpack(tensor::TVMTensor) -> AbstractArray

Convert a TVMTensor to a Julia array via DLPack protocol (zero-copy).

The returned array is a view into the TVMTensor's data. The TVMTensor
must be kept alive as long as the array is used.

# Example
```julia
using DLPack: from_dlpack

tensor = some_tvm_function()
arr = from_dlpack(tensor)
# arr shares memory with tensor
# Keep tensor alive while using arr!
```

# Return Type
- CPU tensors → `Array`
- CUDA tensors → `CuArray` (requires CUDA.jl)
- Other GPU tensors → Appropriate array type (requires corresponding package)

# See also
- [`TVMTensor(arr)`](@ref): Convert Julia array to TVMTensor
"""
function DLPack.from_dlpack(tensor::TVMTensor)
    # Step 1: Export TVM tensor to DLPack (v0.x)
    ret, dlpack_ptr = LibTVMFFI.TVMFFITensorToDLPack(tensor.handle)
    
    if ret != 0
        error("Failed to export TVMTensor to DLPack (ret=$ret)")
    end
    
    if dlpack_ptr == C_NULL
        error("TVMFFITensorToDLPack returned NULL")
    end
    
    # Step 2: Create DLManagedTensor from the pointer
    # DLPack.jl's DLManagedTensor constructor handles this
    managed = DLPack.DLManagedTensor(Ptr{DLPack.DLManagedTensor}(dlpack_ptr))
    
    # Step 3: Use DLPack.jl to wrap as Julia array
    # This handles device type detection and array creation
    arr = Base.unsafe_wrap(managed, tensor)
    
    return arr
end

#=============================================================================
  TVMAny Integration
=============================================================================#

"""
    TVMAny(arr::AbstractArray, ::Val{:dlpack})

Create a TVMAny from a Julia array using DLPack protocol.
This creates a `kTVMFFITensor` (not `kTVMFFIDLTensorPtr`).

The array is kept alive through TVM's reference counting.
"""
function TVMAny(arr::StridedArray, ::Val{:dlpack})
    tensor = TVMTensor(arr)
    return TVMAny(tensor)
end

#=============================================================================
  Module Initialization
=============================================================================#

function _init_dlpack_api()
    # Initialize the deleter function pointer
    _DLPACK_DELETER_CPTR[] = @cfunction(_dlpack_deleter, Cvoid, (Ptr{Cvoid},))
end
