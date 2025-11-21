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
    conversion.jl

Julia ↔ TVM FFI ABI Boundary Layer

This file centralizes all type conversions between Julia and TVM FFI.
Keeping conversions in one place makes it easy to:
- Verify symmetry between to_tvm_any() and from_tvm_any()
- Update the ABI contract when TVM FFI changes
- Audit reference counting for all object types

Design Philosophy (Linus style):
- This is an INTERFACE layer, so centralization is correct
- Like Linux syscalls.h - all ABI entry points in one place
- Not like scattered file operations - those follow their data structures
"""

#=============================================================================
  Julia → TVM FFI Any (Encoding)
=============================================================================#

"""
    to_tvm_any(value) -> LibTVMFFI.TVMFFIAny

Convert Julia value to TVMFFIAny for passing to TVM functions.

# Type Coverage
- POD types: Int64, Float64, Bool
- Device/dtype: DLDevice, DLDataType
- Strings: TVMString, AbstractString
- Objects: TVMFunction, TVMObject, TVMModule, TVMTensor
- Tensors: DLTensorHolder, AbstractArray
- Special: Nothing (null)
"""
function to_tvm_any(value::Int64)
    # POD type - no refcounting
    LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFIInt), 0, reinterpret(UInt64, value))
end

function to_tvm_any(value::Float64)
    # POD type - no refcounting
    LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFIFloat), 0, reinterpret(UInt64, value))
end

function to_tvm_any(value::Bool)
    # POD type - no refcounting
    LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFIBool), 0, UInt64(value))
end

function to_tvm_any(value::DLDevice)
    # POD type - no refcounting
    packed = UInt64(value.device_type) | (UInt64(value.device_id) << 32)
    LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFIDevice), 0, packed)
end

function to_tvm_any(value::DLDataType)
    # POD type - no refcounting
    packed = UInt64(value.code) | (UInt64(value.bits) << 8) | (UInt64(value.lanes) << 16)
    LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFIDataType), 0, packed)
end

function to_tvm_any(value::TVMString)
    # Object type - create new reference
    if value.data.type_index >= Int32(LibTVMFFI.kTVMFFIStaticObjectBegin)
        obj_ptr = reinterpret(LibTVMFFI.TVMFFIObjectHandle, value.data.data)
        if obj_ptr != C_NULL
            LibTVMFFI.TVMFFIObjectIncRef(obj_ptr)
        end
    end
    return value.data
end

function to_tvm_any(value::AbstractString)
    # Convert to TVMString then to Any
    to_tvm_any(TVMString(value))
end

function to_tvm_any(value::TVMFunction)
    # Object type - create new reference
    if value.handle != C_NULL
        LibTVMFFI.TVMFFIObjectIncRef(value.handle)
    end
    LibTVMFFI.TVMFFIAny(
        Int32(LibTVMFFI.kTVMFFIFunction),
        0,
        reinterpret(UInt64, value.handle)
    )
end

function to_tvm_any(value::TVMObject)
    # Generic object - create new reference
    if value.handle != C_NULL
        LibTVMFFI.TVMFFIObjectIncRef(value.handle)
    end
    LibTVMFFI.TVMFFIAny(type_index(value), 0, reinterpret(UInt64, value.handle))
end

function to_tvm_any(value::TVMModule)
    # Module is a wrapper around TVMObject handle
    to_tvm_any(value.handle)
end

function to_tvm_any(::Nothing)
    # Special value - no refcounting
    LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFINone), 0, 0)
end

function to_tvm_any(value::Base.RefValue{DLTensor})
    # Pointer type - no refcounting (borrowed reference)
    LibTVMFFI.TVMFFIAny(
        Int32(LibTVMFFI.kTVMFFIDLTensorPtr),
        0,
        reinterpret(UInt64, Base.unsafe_convert(Ptr{DLTensor}, value))
    )
end

function to_tvm_any(holder::DLTensorHolder)
    # Convert holder to DLTensor pointer
    # Holder keeps data alive, we just borrow the reference
    # Use unsafe_convert which we defined for DLTensorHolder
    tensor_ptr = Base.unsafe_convert(Ptr{DLTensor}, holder)
    LibTVMFFI.TVMFFIAny(
        Int32(LibTVMFFI.kTVMFFIDLTensorPtr),
        0,
        reinterpret(UInt64, tensor_ptr)
    )
end

function to_tvm_any(value::AbstractArray)
    # Auto-convert array to DLTensorHolder then to Any
    # This allows Julia functions to return arrays directly
    holder = from_julia_array(value)
    return to_tvm_any(holder)
end

#=============================================================================
  TVM FFI Any → Julia (Decoding)
=============================================================================#

"""
    from_tvm_any(any::LibTVMFFI.TVMFFIAny; borrowed::Bool = false) -> Any

Convert TVMFFIAny back to Julia value with configurable reference semantics.

# Arguments
- `any`: The TVMFFIAny value to convert
- `borrowed`: Reference borrowing semantics
  - `borrowed=false` (default): C gave us ownership, take it without IncRef
  - `borrowed=true`: C lent us a reference, copy it with IncRef

# Usage Patterns
- **Function returns**: `from_tvm_any(result; borrowed=false)` - C gave us a new reference
- **Callback arguments**: `from_tvm_any(arg; borrowed=true)` - C lent us a borrowed reference

# Examples
```julia
# Pattern 1: Function return (we own the reference)
result = func(x)
value = from_tvm_any(result; borrowed=false)  # Take ownership

# Pattern 2: Callback argument (we borrow the reference)
function my_callback(arg_any)
    value = from_tvm_any(arg_any; borrowed=true)  # Copy reference
end
```

# Note
The parameter name `borrowed` is clearer than `own` because:
- `borrowed=true` → "This is borrowed, I must copy it" (clear!)
- `own=true` → "I own it?" (ambiguous - sounds like taking ownership but actually copies)
"""
function from_tvm_any(any::LibTVMFFI.TVMFFIAny; borrowed::Bool = false)
    type_idx = any.type_index

    if type_idx == Int32(LibTVMFFI.kTVMFFINone)
        return nothing
    elseif type_idx == Int32(LibTVMFFI.kTVMFFIInt)
        return reinterpret(Int64, any.data)
    elseif type_idx == Int32(LibTVMFFI.kTVMFFIBool)
        return any.data != 0
    elseif type_idx == Int32(LibTVMFFI.kTVMFFIFloat)
        return reinterpret(Float64, any.data)
    elseif type_idx == Int32(LibTVMFFI.kTVMFFIDevice)
        device_type = Int32(any.data & 0xFFFFFFFF)
        device_id = Int32((any.data >> 32) & 0xFFFFFFFF)
        return DLDevice(device_type, device_id)
    elseif type_idx == Int32(LibTVMFFI.kTVMFFIDataType)
        code = UInt8(any.data & 0xFF)
        bits = UInt8((any.data >> 8) & 0xFF)
        lanes = UInt16((any.data >> 16) & 0xFFFF)
        return DLDataType(code, bits, lanes)
    elseif type_idx == Int32(LibTVMFFI.kTVMFFIDLTensorPtr)
        # DLTensor pointer - always copy data
        # 
        # WHY COPY? Because DLTensorPtr is just a pointer without ownership info.
        # We don't know:
        # - Who owns the memory (C or Julia)
        # - When it will be freed
        # - Whether it's safe to hold after this call
        #
        # Scenarios:
        # 1. Borrowed (callback arg): C owns data, may free after return → MUST copy
        # 2. Owned (our return): holder is local variable, freed after return → MUST copy
        #
        # For zero-copy, use TVMTensor objects (with refcounting) instead of raw pointers.
        # But that requires different type_index and C understanding object lifecycle.
        #
        # Practical choice: Always copy. Safe and simple. Performance is acceptable
        # for typical callback data sizes.
        tensor_ptr = reinterpret(Ptr{DLTensor}, any.data)
        tensor_ptr == C_NULL && error("NULL DLTensor pointer in from_tvm_any")

        # SAFETY: Read DLTensor metadata (no pointers escaped yet)
        dltensor = unsafe_load(tensor_ptr)

        # Extract shape
        ndim = Int(dltensor.ndim)
        shape = unsafe_wrap(Array, dltensor.shape, ndim) |> copy

        # Determine element type and create result array
        T = dtype_to_julia_type(dltensor.dtype)
        result = Array{T}(undef, shape...)

        # Get strides in ELEMENTS (DLPack standard: strides are in elements, not bytes)
        strides_elem = if dltensor.strides == C_NULL
            # NULL strides → C-contiguous (row-major)
            # DLPack v1.2+ requires explicit strides, but older versions allow NULL
            _compute_c_strides(shape)
        else
            # strides are already in element units per DLPack spec
            unsafe_wrap(Array, dltensor.strides, ndim) |> copy
        end

        # Copy data with correct stride calculation
        data_ptr = Ptr{T}(dltensor.data)
        _copy_strided!(result, data_ptr, shape, strides_elem)

        return result
    elseif type_idx == Int32(LibTVMFFI.kTVMFFISmallStr) ||
           type_idx == Int32(LibTVMFFI.kTVMFFIStr)
        return String(TVMString(any; borrowed = borrowed))
    elseif type_idx == Int32(LibTVMFFI.kTVMFFISmallBytes) ||
           type_idx == Int32(LibTVMFFI.kTVMFFIBytes)
        return Vector{UInt8}(TVMBytes(any; borrowed = borrowed))
    elseif type_idx == Int32(LibTVMFFI.kTVMFFIFunction)
        handle = reinterpret(LibTVMFFI.TVMFFIObjectHandle, any.data)
        return TVMFunction(handle; borrowed = borrowed)
    elseif type_idx == Int32(LibTVMFFI.kTVMFFIError)
        handle = reinterpret(LibTVMFFI.TVMFFIObjectHandle, any.data)
        return TVMError(handle; borrowed = borrowed)
    elseif type_idx == Int32(LibTVMFFI.kTVMFFITensor)
        # Tensor object (with refcounting)
        handle = reinterpret(LibTVMFFI.TVMFFIObjectHandle, any.data)
        return TVMTensor(handle; borrowed = borrowed)
    elseif type_idx == Int32(LibTVMFFI.kTVMFFIModule)
        # Module object
        # TVMModule is a thin wrapper around TVMObject
        handle = reinterpret(LibTVMFFI.TVMFFIObjectHandle, any.data)
        return TVMModule(TVMObject(handle; borrowed = borrowed))
    elseif type_idx == Int32(LibTVMFFI.kTVMFFIOpaquePtr)
        # Opaque pointer - just return as Ptr{Cvoid}
        return reinterpret(Ptr{Cvoid}, any.data)
    elseif type_idx >= Int32(LibTVMFFI.kTVMFFIStaticObjectBegin)
        # Generic object (covers Array, Map, Shape, etc.)
        handle = reinterpret(LibTVMFFI.TVMFFIObjectHandle, any.data)
        return TVMObject(handle; borrowed = borrowed)
    else
        error("Unsupported type index for conversion: $type_idx (add support if needed)")
    end
end

#=============================================================================
  Helper Functions (used by from_tvm_any)
=============================================================================#

"""
Compute C-contiguous (row-major) strides from shape.
"""
function _compute_c_strides(shape)
    ndim = length(shape)
    strides = zeros(Int64, ndim)
    if ndim > 0
        strides[end] = 1
        for i in (ndim - 1):-1:1
            strides[i] = strides[i + 1] * shape[i + 1]
        end
    end
    return strides
end

"""
Copy data with arbitrary strides.
Handles both C-order (row-major) and Fortran-order (column-major) correctly.
"""
function _copy_strided!(dst::Array{T}, src_ptr::Ptr{T}, shape, strides_elem) where {T}
    ndim = length(shape)
    indices = ones(Int, ndim)
    total = prod(shape)

    for linear_idx in 1:total
        # Compute source offset using strides
        src_offset = sum((indices[i] - 1) * strides_elem[i] for i in 1:ndim)
        dst[linear_idx] = unsafe_load(src_ptr, src_offset + 1)

        # Increment indices (column-major for Julia)
        for d in 1:ndim
            if indices[d] < shape[d]
                indices[d] += 1
                break
            else
                indices[d] = 1
            end
        end
    end
end
