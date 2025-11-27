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
    any.jl

TVM FFI Any/AnyView Types - Ownership-aware value containers.

Design inspired by Rust's `Any` and `AnyView<'a>`:
- `TVMAnyView`: Borrowed reference, does NOT manage lifetime
- `TVMAny`: Owned value, manages reference count via finalizer

This separation makes ownership explicit at the type level,
preventing accidental reference count errors.
"""

using .LibTVMFFI

#=============================================================================
  TVMAnyView - Borrowed Reference (No Lifetime Management)
=============================================================================#

"""
    TVMAnyView

A borrowed view into a TVM value. Does NOT manage reference counts.

Use this type when:
- Receiving callback arguments from C (C owns the reference)
- Temporarily accessing values without taking ownership

# Warning
The view is only valid while the underlying C reference is alive.
Do NOT store `TVMAnyView` beyond the current scope.

# Example
```julia
function my_callback(args_ptr::Ptr{LibTVMFFI.TVMFFIAny}, num_args::Int)
    for i in 1:num_args
        # Create view - does not IncRef
        view = TVMAnyView(unsafe_load(args_ptr, i))
        
        # Convert to owned value if needed beyond this scope
        owned = TVMAny(view)
    end
end
```
"""
struct TVMAnyView
    data::LibTVMFFI.TVMFFIAny
end

"""
    type_index(view::TVMAnyView) -> Int32

Get the type index of the value.
"""
type_index(view::TVMAnyView) = view.data.type_index

"""
    is_object(view::TVMAnyView) -> Bool

Check if the value is a reference-counted object.
"""
function is_object(view::TVMAnyView)
    view.data.type_index >= Int32(LibTVMFFI.kTVMFFIStaticObjectBegin)
end

"""
    is_none(view::TVMAnyView) -> Bool

Check if the value is None/null.
"""
is_none(view::TVMAnyView) = view.data.type_index == Int32(LibTVMFFI.kTVMFFINone)

#=============================================================================
  TVMAny - Owned Value (Manages Reference Count)
=============================================================================#

"""
    TVMAny

An owned TVM value. Manages reference count for object types.

Use this type when:
- Receiving function return values (C transfers ownership)
- Storing values beyond the current scope
- Converting from `TVMAnyView` to keep the value alive

# Lifecycle
- Construction: Does NOT IncRef (assumes ownership transferred)
- Destruction: DecRef for object types via finalizer

# Example
```julia
# Function return - take ownership directly
result_any = TVMAny(raw_result)

# From borrowed view - uses TVMFFIAnyViewToOwnedAny
owned = TVMAny(view::TVMAnyView)
```
"""
mutable struct TVMAny
    data::LibTVMFFI.TVMFFIAny

    # Constructor from raw TVMFFIAny - takes ownership (no IncRef)
    function TVMAny(raw::LibTVMFFI.TVMFFIAny)
        any = new(raw)

        # Register finalizer for object types
        if raw.type_index >= Int32(LibTVMFFI.kTVMFFIStaticObjectBegin)
            finalizer(any) do a
                obj_ptr = reinterpret(LibTVMFFI.TVMFFIObjectHandle, a.data.data)
                if obj_ptr != C_NULL
                    LibTVMFFI.TVMFFIObjectDecRef(obj_ptr)
                end
            end
        end

        return any
    end

    # Inner constructor for creating without finalizer (internal use)
    function TVMAny(raw::LibTVMFFI.TVMFFIAny, ::Val{:no_finalizer})
        new(raw)
    end
end

"""
    TVMAny(view::TVMAnyView)

Convert a borrowed view to an owned value using the official C API.

This properly increments the reference count for object types,
making it safe to hold the value beyond the view's lifetime.
"""
function TVMAny(view::TVMAnyView)
    ret, owned_raw = LibTVMFFI.TVMFFIAnyViewToOwnedAny(view.data)
    if ret != 0
        error("Failed to convert AnyView to owned Any (ret=$ret)")
    end
    return TVMAny(owned_raw)
end

"""
    type_index(any::TVMAny) -> Int32

Get the type index of the value.
"""
type_index(any::TVMAny) = any.data.type_index

"""
    is_object(any::TVMAny) -> Bool

Check if the value is a reference-counted object.
"""
is_object(any::TVMAny) = any.data.type_index >= Int32(LibTVMFFI.kTVMFFIStaticObjectBegin)

"""
    is_none(any::TVMAny) -> Bool

Check if the value is None/null.
"""
is_none(any::TVMAny) = any.data.type_index == Int32(LibTVMFFI.kTVMFFINone)

#=============================================================================
  Conversion Functions - Type-Safe Extraction
=============================================================================#

"""
    take_value(any::TVMAny) -> Any

Extract the Julia value from an owned TVMAny, consuming ownership.

For object types, the returned wrapper (TVMFunction, TVMObject, etc.)
takes over reference management. The TVMAny is invalidated after this call.

# Example
```julia
result = func(x)  # Returns TVMAny
value = take_value(result)  # Extracts and transfers ownership
# result is now invalidated
```

# Implementation Note
After take_value, the TVMAny's data is cleared to prevent double-free.
The finalizer will no longer DecRef the object.
"""
function take_value end

"""
    copy_value(view::TVMAnyView) -> Any

Extract the Julia value from a borrowed view, copying if necessary.

For object types, this increments the reference count so the
returned value is independent of the original view's lifetime.

# Example
```julia
function callback(view::TVMAnyView)
    value = copy_value(view)  # Safe to use after callback returns
    return value
end
```
"""
function copy_value end

# Implementation note: take_value/copy_value are implemented in conversion.jl
# after all types (TVMFunction, TVMObject, etc.) are defined.

#=============================================================================
  Julia Value → TVMAny Constructors (Forward Declarations)

  Implemented in conversion.jl after all types are defined.
=============================================================================#

# POD types - no reference counting needed
# TVMAny(::Int64)
# TVMAny(::Float64) 
# TVMAny(::Bool)
# TVMAny(::Nothing)
# TVMAny(::DLDevice)
# TVMAny(::DLDataType)

# Object types - IncRef + finalizer manages DecRef
# TVMAny(::TVMString)
# TVMAny(::TVMFunction)
# TVMAny(::TVMObject)
# TVMAny(::TVMModule)

# Array types
# TVMAny(::TensorView)
# TVMAny(::AbstractArray)

# String convenience
# TVMAny(::AbstractString)

#=============================================================================
  Helper Functions
=============================================================================#

"""
    raw_data(any::TVMAny) -> LibTVMFFI.TVMFFIAny

Get the raw TVMFFIAny data (internal use only).
"""
raw_data(any::TVMAny) = any.data

"""
    transfer_ownership!(any::TVMAny) -> LibTVMFFI.TVMFFIAny

Transfer ownership out of the TVMAny, returning the raw data.
After this call, the finalizer will NOT DecRef (data is cleared for objects).

Use this when passing values to C code that takes ownership,
such as writing callback return values.

# Example
```julia
# In callback return path:
result_any = TVMAny(some_object)  # IncRef happened
raw = transfer_ownership!(result_any)  # Prevents finalizer DecRef
unsafe_store!(ret_ptr, raw)  # C now owns the reference
# result_any finalizer runs but does nothing (data cleared)
```
"""
function transfer_ownership!(any::TVMAny)
    data = any.data
    # Clear data for object types so finalizer won't DecRef
    if data.type_index >= Int32(LibTVMFFI.kTVMFFIStaticObjectBegin)
        any.data = LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFINone), 0, 0)
    end
    return data
end

"""
    raw_data(view::TVMAnyView) -> LibTVMFFI.TVMFFIAny

Get the raw TVMFFIAny data (internal use only).
"""
raw_data(view::TVMAnyView) = view.data

# Display
function Base.show(io::IO, view::TVMAnyView)
    print(io, "TVMAnyView(type_index=", view.data.type_index, ")")
end

function Base.show(io::IO, any::TVMAny)
    print(io, "TVMAny(type_index=", any.data.type_index, ")")
end
