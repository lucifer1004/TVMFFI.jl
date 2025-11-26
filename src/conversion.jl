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
- Verify symmetry between TVMAny(value) and take_value()/copy_value()
- Update the ABI contract when TVM FFI changes
- Audit reference counting for all object types

Design Philosophy (Linus style):
- This is an INTERFACE layer, so centralization is correct
- Like Linux syscalls.h - all ABI entry points in one place
- Not like scattered file operations - those follow their data structures
"""

#=============================================================================
  Julia → TVMAny Constructors
  
  Creates managed TVMAny from Julia values.
  - POD types: No reference counting needed
  - Object types: IncRef + finalizer manages DecRef
=============================================================================#

# ---- POD Types (no reference counting) ----

"""
    TVMAny(value::Int64)

Create a TVMAny from an integer value.
"""
function TVMAny(value::Int64)
    raw = LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFIInt), 0, reinterpret(UInt64, value))
    TVMAny(raw, Val(:no_finalizer))
end

"""
    TVMAny(value::Float64)

Create a TVMAny from a floating-point value.
"""
function TVMAny(value::Float64)
    raw = LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFIFloat), 0, reinterpret(UInt64, value))
    TVMAny(raw, Val(:no_finalizer))
end

"""
    TVMAny(value::Bool)

Create a TVMAny from a boolean value.
"""
function TVMAny(value::Bool)
    raw = LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFIBool), 0, UInt64(value))
    TVMAny(raw, Val(:no_finalizer))
end

"""
    TVMAny(::Nothing)

Create a TVMAny representing None/null.
"""
function TVMAny(::Nothing)
    raw = LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFINone), 0, 0)
    TVMAny(raw, Val(:no_finalizer))
end

"""
    TVMAny(value::DLDevice)

Create a TVMAny from a device descriptor.
"""
function TVMAny(value::DLDevice)
    packed = UInt64(value.device_type) | (UInt64(value.device_id) << 32)
    raw = LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFIDevice), 0, packed)
    TVMAny(raw, Val(:no_finalizer))
end

"""
    TVMAny(value::DLDataType)

Create a TVMAny from a data type descriptor.
"""
function TVMAny(value::DLDataType)
    packed = UInt64(value.code) | (UInt64(value.bits) << 8) | (UInt64(value.lanes) << 16)
    raw = LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFIDataType), 0, packed)
    TVMAny(raw, Val(:no_finalizer))
end

# ---- Object Types (IncRef + finalizer DecRef) ----

"""
    TVMAny(value::TVMString)

Create a TVMAny from a TVM string. Increments reference count.
"""
function TVMAny(value::TVMString)
    # IncRef for object types
    if value.data.type_index >= Int32(LibTVMFFI.kTVMFFIStaticObjectBegin)
        obj_ptr = reinterpret(LibTVMFFI.TVMFFIObjectHandle, value.data.data)
        if obj_ptr != C_NULL
            LibTVMFFI.TVMFFIObjectIncRef(obj_ptr)
        end
    end
    # Use main constructor which registers finalizer for objects
    TVMAny(value.data)
end

"""
    TVMAny(value::AbstractString)

Create a TVMAny from a Julia string (converts to TVMString first).
"""
function TVMAny(value::AbstractString)
    TVMAny(TVMString(value))
end

"""
    TVMAny(value::TVMFunction)

Create a TVMAny from a TVM function. Increments reference count.
"""
function TVMAny(value::TVMFunction)
    if value.handle != C_NULL
        LibTVMFFI.TVMFFIObjectIncRef(value.handle)
    end
    raw = LibTVMFFI.TVMFFIAny(
        Int32(LibTVMFFI.kTVMFFIFunction),
        0,
        reinterpret(UInt64, value.handle)
    )
    TVMAny(raw)  # Main constructor registers finalizer
end

"""
    TVMAny(value::TVMObject)

Create a TVMAny from a generic TVM object. Increments reference count.
"""
function TVMAny(value::TVMObject)
    if value.handle != C_NULL
        LibTVMFFI.TVMFFIObjectIncRef(value.handle)
    end
    raw = LibTVMFFI.TVMFFIAny(type_index(value), 0, reinterpret(UInt64, value.handle))
    TVMAny(raw)  # Main constructor registers finalizer
end

"""
    TVMAny(value::TVMModule)

Create a TVMAny from a TVM module.
"""
function TVMAny(value::TVMModule)
    TVMAny(value.handle)
end

# ---- TensorView Types (pointer-based, no refcounting) ----

"""
    TVMAny(value::Base.RefValue{DLTensor})

Create a TVMAny from a DLTensor reference.
"""
function TVMAny(value::Base.RefValue{DLTensor})
    raw = LibTVMFFI.TVMFFIAny(
        Int32(LibTVMFFI.kTVMFFIDLTensorPtr),
        0,
        reinterpret(UInt64, Base.unsafe_convert(Ptr{DLTensor}, value))
    )
    TVMAny(raw, Val(:no_finalizer))
end

"""
    TVMAny(view::TensorView)

Create a TVMAny from a TensorView.

Note: The TensorView must be kept alive while this TVMAny is used!
"""
function TVMAny(view::TensorView)
    tensor_ptr = Base.unsafe_convert(Ptr{DLTensor}, view)
    raw = LibTVMFFI.TVMFFIAny(
        Int32(LibTVMFFI.kTVMFFIDLTensorPtr),
        0,
        reinterpret(UInt64, tensor_ptr)
    )
    TVMAny(raw, Val(:no_finalizer))
end

"""
    TVMAny(value::AbstractArray)

Create a TVMAny from a Julia array (converts to TensorView first).

Note: Returns a tuple (any, view) - the view must be kept alive!
"""
function TVMAny(value::AbstractArray)
    view = TensorView(value)
    any = TVMAny(view)
    return any, view  # Caller must GC.@preserve view
end

#=============================================================================
  TVM FFI Any → Julia (Decoding)
  
  Two main APIs:
  - take_value(any::TVMAny) - Extract from owned value (no IncRef needed)
  - copy_value(view::TVMAnyView) - Extract from borrowed view (needs copy/IncRef)
=============================================================================#

"""
    take_value(any::TVMAny) -> Any

Extract the Julia value from an owned TVMAny, consuming ownership.

The TVMAny already owns the reference (or it's a POD type).
For object types, the returned wrapper takes over reference management.
After this call, the TVMAny is invalidated and will not DecRef.

# Usage
```julia
# Function return - already owned
result_any = TVMAny(raw_result)
value = take_value(result_any)
# result_any is now invalid
```

# Warning
After calling take_value, the TVMAny is invalidated. Do not use it again.
"""
function take_value(any::TVMAny)
    data = any.data
    
    # Invalidate the TVMAny to prevent finalizer from DecRef
    # This transfers ownership to the returned wrapper
    if data.type_index >= Int32(LibTVMFFI.kTVMFFIStaticObjectBegin)
        # Clear the data so finalizer won't DecRef
        any.data = LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFINone), 0, 0)
    end
    
    _extract_value(data, false)
end

"""
    copy_value(view::TVMAnyView) -> Any

Extract the Julia value from a borrowed view, copying if necessary.

For object types, this uses the official `TVMFFIAnyViewToOwnedAny` C API
to increment the reference count, ensuring the returned value is
independent of the original view's lifetime.

# Usage
```julia
# Callback argument - borrowed, need to copy
function my_callback(view::TVMAnyView)
    value = copy_value(view)  # Safe to use after callback returns
    return value
end
```
"""
function copy_value(view::TVMAnyView)
    # Use official C API to convert view → owned
    ret, owned_raw = LibTVMFFI.TVMFFIAnyViewToOwnedAny(view.data)
    if ret != 0
        error("Failed to copy AnyView to owned Any (ret=$ret)")
    end
    # owned_raw is now IncRef'd, extract without additional IncRef
    _extract_value(owned_raw, false)
end

"""
Internal implementation: extract Julia value from raw TVMFFIAny.

# Arguments
- `any`: Raw TVMFFIAny data
- `borrowed`: If true, IncRef object types to copy the reference
"""
function _extract_value(any::LibTVMFFI.TVMFFIAny, borrowed::Bool)
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
        # 2. Owned (our return): view is local variable, freed after return → MUST copy
        #
        # For zero-copy, use TVMTensor objects (with refcounting) instead of raw pointers.
        # But that requires different type_index and C understanding object lifecycle.
        #
        # Practical choice: Always copy. Safe and simple. Performance is acceptable
        # for typical callback data sizes.
        tensor_ptr = reinterpret(Ptr{DLTensor}, any.data)
        tensor_ptr == C_NULL && error("NULL DLTensor pointer in _extract_value")

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
