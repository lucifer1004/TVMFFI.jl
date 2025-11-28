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

Centralizes all type conversions between Julia and TVM FFI:
- `TVMAny(value)`: Julia → TVM
- `take_value(any)`: TVM → Julia (owned)
- `copy_value(view)`: TVM → Julia (borrowed)
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

# Support other integer types - TVM FFI ABI stores all integers as kTVMFFIInt + int64
# This matches Rust/Python implementations where all integer types are converted to i64
# See: tvm-ffi/rust/tvm-ffi/src/type_traits.rs impl_any_compatible_for_int! macro
TVMAny(value::Int32) = TVMAny(Int64(value))
TVMAny(value::Int16) = TVMAny(Int64(value))
TVMAny(value::Int8) = TVMAny(Int64(value))
TVMAny(value::UInt64) = TVMAny(reinterpret(Int64, value))
TVMAny(value::UInt32) = TVMAny(Int64(value))
TVMAny(value::UInt16) = TVMAny(Int64(value))
TVMAny(value::UInt8) = TVMAny(Int64(value))

"""
    TVMAny(value::Float64)

Create a TVMAny from a 64-bit floating-point value.
"""
function TVMAny(value::Float64)
    raw = LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFIFloat), 0, reinterpret(UInt64, value))
    TVMAny(raw, Val(:no_finalizer))
end

"""
    TVMAny(value::Float32)

Create a TVMAny from a 32-bit floating-point value.
Note: TVM FFI uses 64-bit floats internally, so Float32 is promoted to Float64.
"""
function TVMAny(value::Float32)
    TVMAny(Float64(value))
end

"""
    TVMAny(value::Float16)

Create a TVMAny from a 16-bit floating-point value.
Note: TVM FFI uses 64-bit floats internally, so Float16 is promoted to Float64.
"""
function TVMAny(value::Float16)
    TVMAny(Float64(value))
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

"""
    TVMAny(value::TVMTensor)

Create a TVMAny from a TVM tensor. Increments reference count.
"""
function TVMAny(value::TVMTensor)
    handle = value.handle
    if handle == C_NULL
        return TVMAny(LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFINone), 0, 0))
    end

    # IncRef: TVMAny will have its own reference, TVMTensor keeps its reference
    # Both will DecRef independently via their finalizers
    LibTVMFFI.TVMFFIObjectIncRef(handle)

    # Create TVMAny - main constructor registers finalizer that will DecRef
    raw = LibTVMFFI.TVMFFIAny(
        Int32(LibTVMFFI.kTVMFFITensor),
        0,
        reinterpret(UInt64, handle)
    )
    TVMAny(raw)
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
        # DLTensor pointer - copy data for CPU, error for GPU
        # 
        # WHY COPY? Because DLTensorPtr is just a pointer without ownership info.
        # We don't know:
        # - Who owns the memory (C or Julia)
        # - When it will be freed
        # - Whether it's safe to hold after this call
        #
        # For zero-copy, use TVMTensor objects (with refcounting) instead of raw pointers.
        tensor_ptr = reinterpret(Ptr{DLTensor}, any.data)
        tensor_ptr == C_NULL && error("NULL DLTensor pointer in _extract_value")

        # SAFETY: Read DLTensor metadata (no pointers escaped yet)
        dltensor = unsafe_load(tensor_ptr)

        # Check device type - GPU tensors cannot be copied with simple memcpy
        device_type = dltensor.device.device_type
        if device_type != Int32(LibTVMFFI.kDLCPU)
            error("GPU DLTensor (device_type=$device_type) in callback requires kTVMFFITensor type " *
                  "(with reference counting) for proper GPU memory handling. " *
                  "Raw DLTensorPtr only supports CPU arrays.")
        end

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
        # Tensor object (with refcounting) - zero-copy via DLPack
        handle = reinterpret(LibTVMFFI.TVMFFIObjectHandle, any.data)
        tensor = TVMTensor(handle; borrowed = borrowed)
        # Convert to Julia array via DLPack (zero-copy)
        # The array holds a reference to tensor, keeping it alive
        return from_dlpack(tensor)
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
