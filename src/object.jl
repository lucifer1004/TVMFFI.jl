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

using .LibTVMFFI

# Cache for type indices
const _type_key2index = Dict{String, Int32}()
const _type_index2type = Dict{Int32, Type}()
const _object_lock = ReentrantLock()

function _init_object_api()
    # Register basic types
    # TODO: Add basic types if needed
end

"""
    TVMObject

Base wrapper for TVM FFI object handles with automatic memory management.
"""
mutable struct TVMObject
    handle::LibTVMFFI.TVMFFIObjectHandle

    """
        TVMObject(handle; borrowed)

    Create a TVMObject from a raw handle. Internal API - users should not call directly.

    # Arguments
    - `handle`: The raw object handle
    - `borrowed`: Reference semantics (REQUIRED - no default to prevent misuse)
      - `borrowed=true`: Borrowed reference, increment refcount
      - `borrowed=false`: Owned reference, take without IncRef (C gave us ownership)
    """
    function TVMObject(handle::LibTVMFFI.TVMFFIObjectHandle; borrowed::Bool)
        if handle == C_NULL
            error("Cannot create TVMObject from NULL handle")
        end

        # Copy reference if borrowed
        if borrowed
            LibTVMFFI.TVMFFIObjectIncRef(handle)
        end

        obj = new(handle)

        # Finalizer to decrease ref count when GC collects this
        finalizer(obj) do o
            if o.handle != C_NULL
                LibTVMFFI.TVMFFIObjectDecRef(o.handle)
            end
        end

        return obj
    end
end

"""
    type_index(obj::TVMObject) -> Int32

Get the type index of an object.
"""
function type_index(obj::TVMObject)
    LibTVMFFI.TVMFFIObjectGetTypeIndex(obj.handle)
end

"""
    is_type(obj::TVMObject, idx::LibTVMFFI.TVMFFITypeIndex) -> Bool

Check if an object has a specific type index.
"""
function is_type(obj::TVMObject, idx::LibTVMFFI.TVMFFITypeIndex)
    type_index(obj) == Int32(idx)
end

Base.show(io::IO, obj::TVMObject) = print(io, "TVMObject(type_index=", type_index(obj), ")")

"""
    register_object(type_key::String, T::Type; parent_type_index::Int32 = Int32(64))

Register a Julia type as a TVM object type.

# Arguments
- `type_key`: The type key defined in C++ (e.g., "tvm.RelayExpr")
- `T`: The Julia type to map to this type key
- `parent_type_index`: Parent type index (default: 64 = ffi.Object)
"""
function register_object(type_key::String, T::Type; parent_type_index::Int32 = Int32(64))
    # Use GetOrAllocIndex to register new types if they don't exist
    # New object types must have ffi.Object (index 64) as parent at minimum
    local idx
    GC.@preserve type_key begin
        key_bytes = LibTVMFFI.TVMFFIByteArray(
            Ptr{UInt8}(pointer(type_key)), UInt(sizeof(type_key))
        )
        idx = LibTVMFFI.TVMFFITypeGetOrAllocIndex(
            key_bytes,
            Int32(-1),  # static_type_index: -1 for dynamic types
            Int32(1),   # type_depth: 1 (parent is at depth 0)
            Int32(0),   # num_child_slots: 0 (no pre-allocated child slots)
            Int32(0),   # child_slots_can_overflow: 0 (fixed)
            parent_type_index  # parent is ffi.Object by default
        )
    end

    lock(_object_lock) do
        _type_key2index[type_key] = idx
        _type_index2type[idx] = T
    end

    return idx
end

"""
    get_type_index(type_key::String) -> Int32

Get the type index for a given type key.
"""
function get_type_index(type_key::String)
    lock(_object_lock) do
        if haskey(_type_key2index, type_key)
            return _type_key2index[type_key]
        end
    end

    # Not in cache, query C API
    local ret, idx
    GC.@preserve type_key begin
        key_bytes = LibTVMFFI.TVMFFIByteArray(
            Ptr{UInt8}(pointer(type_key)), UInt(sizeof(type_key))
        )
        ret, idx = LibTVMFFI.TVMFFITypeKeyToIndex(key_bytes)
    end

    if ret != 0
        error("Type key '$type_key' not found in TVM registry")
    end

    lock(_object_lock) do
        _type_key2index[type_key] = idx
    end

    return idx
end

"""
    type_key(T::Type) -> String

Get the type key for a registered Julia type. Returns nothing if not registered.
"""
function type_key end

"""
    @register_object type_key struct TypeName [<: ParentType] ... end

Register a Julia struct as a TVM object type with automatic memory management.

This macro generates:
1. A mutable struct with a `handle` field for the TVM object handle
2. A constructor that properly manages reference counting
3. A finalizer for automatic cleanup
4. Type index methods for runtime type queries

# Arguments
- `type_key`: The TVM type key (e.g., "testing.MyObject")
- The struct definition (fields are for documentation only; actual field access
  requires TVM reflection API support in C++)

# Examples

```julia
# Basic usage - wrap existing TVM type
@register_object "ffi.Module" struct Module end

# With parent type annotation (for documentation)
@register_object "testing.TestObject" struct TestObject <: TVMObjectBase
    v_i64::Int64   # Field declaration (actual access via TVM)
    v_f64::Float64
end

# After registration, create instances from handles:
obj = TestObject(handle; borrowed=false)  # Take ownership
obj = TestObject(handle; borrowed=true)   # Copy reference
```

# Notes
- The type key must be registered on the C++ side first
- Field declarations are informational; actual field access depends on
  TVM's reflection API being available for that type
- For types with `__ffi_init__`, use `get_global_func` to call constructors

See also: [`register_object`](@ref), [`get_type_index`](@ref), [`type_index`](@ref)
"""
# Global cache for registered type indices (type -> index)
const _registered_type_indices = Dict{DataType, Int32}()

macro register_object(type_key, struct_def)
    # Parse the struct definition
    if !Meta.isexpr(struct_def, :struct)
        error("@register_object expects a struct definition")
    end

    is_mutable = struct_def.args[1]
    type_expr = struct_def.args[2]
    body = struct_def.args[3]

    # Extract type name and parent type
    local type_name, parent_type
    if Meta.isexpr(type_expr, :<:)
        type_name = type_expr.args[1]
        parent_type = type_expr.args[2]
    else
        type_name = type_expr
        parent_type = nothing
    end

    # Extract field declarations from body (for documentation/future use)
    field_decls = []
    for expr in body.args
        if expr isa LineNumberNode
            continue
        elseif Meta.isexpr(expr, :(::))
            push!(field_decls, (name=expr.args[1], type=expr.args[2]))
        elseif expr isa Symbol
            push!(field_decls, (name=expr, type=:Any))
        end
    end

    type_key_str = type_key  # Save for use in generated code

    # Generate the struct and registration code
    quote
        # The actual mutable struct with handle
        mutable struct $(esc(type_name))
            handle::LibTVMFFI.TVMFFIObjectHandle

            """
                $($(string(type_name)))(handle; borrowed)

            Create a $($(string(type_name))) from a raw handle.

            # Arguments
            - `handle`: The raw TVM object handle
            - `borrowed`: Reference semantics (REQUIRED)
              - `borrowed=true`: Borrowed reference, increment refcount
              - `borrowed=false`: Owned reference, take without IncRef
            """
            function $(esc(type_name))(
                handle::LibTVMFFI.TVMFFIObjectHandle;
                borrowed::Bool
            )
                if handle == C_NULL
                    error("Cannot create " * $(string(type_name)) * " from NULL handle")
                end

                # Copy reference if borrowed
                if borrowed
                    LibTVMFFI.TVMFFIObjectIncRef(handle)
                end

                obj = new(handle)

                # Finalizer to decrease ref count when GC collects this
                finalizer(obj) do o
                    if o.handle != C_NULL
                        LibTVMFFI.TVMFFIObjectDecRef(o.handle)
                    end
                end

                return obj
            end
        end

        # Register the type and cache the type index
        let type_key_str = $(esc(type_key))
            local idx
            try
                idx = get_type_index(type_key_str)
            catch
                # Type not registered in C++, register it now
                idx = register_object(type_key_str, $(esc(type_name)))
            end
            # Update caches
            lock(_object_lock) do
                _type_index2type[idx] = $(esc(type_name))
                _registered_type_indices[$(esc(type_name))] = idx
            end
        end

        # Type index methods - look up from cache
        function TVMFFI.type_index(obj::$(esc(type_name)))
            LibTVMFFI.TVMFFIObjectGetTypeIndex(obj.handle)
        end

        function TVMFFI.type_index(::Type{$(esc(type_name))})
            _registered_type_indices[$(esc(type_name))]
        end

        # Type key method
        TVMFFI.type_key(::Type{$(esc(type_name))}) = $(esc(type_key))

        # Show method
        function Base.show(io::IO, obj::$(esc(type_name)))
            print(io, $(string(type_name)), "(type_index=", TVMFFI.type_index(obj), ")")
        end

        # Return the type
        $(esc(type_name))
    end
end

"""
    @register_object_simple type_key TypeName

A simplified version of @register_object that only registers the type
without creating a new struct. Use when you want to manually define the struct.

# Example
```julia
mutable struct MyCustomObject
    handle::LibTVMFFI.TVMFFIObjectHandle
    cached_value::Int  # Custom cached field

    function MyCustomObject(handle; borrowed::Bool)
        # Custom constructor logic
        ...
    end
end

@register_object_simple "my.CustomObject" MyCustomObject
```
"""
macro register_object_simple(type_key, type_name)
    quote
        # Register the type and cache the type index
        let type_key_str = $(esc(type_key))
            local idx
            try
                idx = get_type_index(type_key_str)
            catch
                idx = register_object(type_key_str, $(esc(type_name)))
            end
            lock(_object_lock) do
                _type_index2type[idx] = $(esc(type_name))
                _registered_type_indices[$(esc(type_name))] = idx
            end
        end

        # Type index methods
        function TVMFFI.type_index(obj::$(esc(type_name)))
            LibTVMFFI.TVMFFIObjectGetTypeIndex(obj.handle)
        end

        function TVMFFI.type_index(::Type{$(esc(type_name))})
            _registered_type_indices[$(esc(type_name))]
        end

        # Type key method
        TVMFFI.type_key(::Type{$(esc(type_name))}) = $(esc(type_key))

        # Return the type index
        _registered_type_indices[$(esc(type_name))]
    end
end
