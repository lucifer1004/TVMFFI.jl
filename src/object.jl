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
    type_index(obj) -> Int32
    type_index(T::Type) -> Int32

Get the runtime type index of an object or the registered type index for a type.

For objects, returns the actual runtime type index from the C++ object header.
For types, returns the type index allocated during registration.

# Examples
```julia
obj = TestCxxClassBase(42, 10)
type_index(obj)              # Runtime index (e.g., 133)
type_index(TestCxxClassBase) # Same as above for registered types
type_index(TVMFunction)      # Built-in type index (68)
```
"""
function type_index end

type_index(obj::TVMObject) = LibTVMFFI.TVMFFIObjectGetTypeIndex(obj.handle)

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

#------------------------------------------------------------
# Section: Reflection API
#------------------------------------------------------------

"""
    get_type_info(type_index::Int32) -> Union{LibTVMFFI.TVMFFITypeInfo, Nothing}

Get the type information for a given type index.
Returns nothing if the type index is invalid.
"""
function get_type_info(type_index::Int32)
    ptr = LibTVMFFI.TVMFFIGetTypeInfo(type_index)
    if ptr == C_NULL
        return nothing
    end
    return unsafe_load(ptr)
end

"""
    get_type_info(type_key::String) -> Union{LibTVMFFI.TVMFFITypeInfo, Nothing}

Get the type information for a given type key.
"""
function get_type_info(type_key::String)
    try
        idx = get_type_index(type_key)
        return get_type_info(idx)
    catch
        return nothing
    end
end

"""
    FieldInfo

Julia wrapper for TVMFFIFieldInfo with convenient accessors.
"""
struct FieldInfo
    name::String
    doc::String
    metadata::String
    flags::Int64
    is_writable::Bool
    has_default::Bool
    getter::Ptr{Cvoid}
    setter::Ptr{Cvoid}
    static_type_index::Int32
    offset::Int64  # Byte offset from object start
end

function FieldInfo(info::LibTVMFFI.TVMFFIFieldInfo)
    name = unsafe_string(info.name.data, info.name.size)
    doc = info.doc.data == C_NULL ? "" : unsafe_string(info.doc.data, info.doc.size)
    metadata = info.metadata.data == C_NULL ? "" : unsafe_string(info.metadata.data, info.metadata.size)

    FieldInfo(
        name, doc, metadata, info.flags,
        (info.flags & LibTVMFFI.kTVMFFIFieldFlagBitMaskWritable) != 0,
        (info.flags & LibTVMFFI.kTVMFFIFieldFlagBitMaskHasDefault) != 0,
        info.getter, info.setter, info.field_static_type_index,
        info.offset
    )
end

"""
    MethodInfo

Julia wrapper for TVMFFIMethodInfo with convenient accessors.
"""
struct MethodInfo
    name::String
    doc::String
    metadata::String
    flags::Int64
    is_static::Bool
    method_handle::LibTVMFFI.TVMFFIObjectHandle  # Stored as handle, converted to TVMFunction on call
end

function MethodInfo(info::LibTVMFFI.TVMFFIMethodInfo)
    name = unsafe_string(info.name.data, info.name.size)
    doc = info.doc.data == C_NULL ? "" : unsafe_string(info.doc.data, info.doc.size)
    metadata = info.metadata.data == C_NULL ? "" : unsafe_string(info.metadata.data, info.metadata.size)

    # Extract the method handle from TVMFFIAny
    method_handle = reinterpret(LibTVMFFI.TVMFFIObjectHandle, info.method.data)
    # IncRef because we're keeping a reference
    if method_handle != C_NULL
        LibTVMFFI.TVMFFIObjectIncRef(method_handle)
    end

    MethodInfo(
        name, doc, metadata, info.flags,
        (info.flags & LibTVMFFI.kTVMFFIMethodFlagBitMaskStatic) != 0,
        method_handle
    )
end

# get_method_function is defined in function.jl after TVMFunction is available

"""
    get_fields(type_info::LibTVMFFI.TVMFFITypeInfo) -> Vector{FieldInfo}

Get all fields defined for a type.
"""
function get_fields(type_info::LibTVMFFI.TVMFFITypeInfo)
    fields = FieldInfo[]
    if type_info.num_fields > 0 && type_info.fields != C_NULL
        for i in 1:type_info.num_fields
            field_ptr = type_info.fields + (i - 1) * sizeof(LibTVMFFI.TVMFFIFieldInfo)
            field = unsafe_load(Ptr{LibTVMFFI.TVMFFIFieldInfo}(field_ptr))
            push!(fields, FieldInfo(field))
        end
    end
    return fields
end

"""
    get_methods(type_info::LibTVMFFI.TVMFFITypeInfo) -> Vector{MethodInfo}

Get all methods defined for a type.
"""
function get_methods(type_info::LibTVMFFI.TVMFFITypeInfo)
    methods = MethodInfo[]
    if type_info.num_methods > 0 && type_info.methods != C_NULL
        for i in 1:type_info.num_methods
            method_ptr = type_info.methods + (i - 1) * sizeof(LibTVMFFI.TVMFFIMethodInfo)
            method = unsafe_load(Ptr{LibTVMFFI.TVMFFIMethodInfo}(method_ptr))
            push!(methods, MethodInfo(method))
        end
    end
    return methods
end

"""
    get_field_value(obj, field::FieldInfo) -> Any

Read a field value from an object using the reflection getter.
"""
function get_field_value(obj, field::FieldInfo)
    if field.getter == C_NULL
        error("Field '$(field.name)' has no getter")
    end

    # Get pointer to the object
    handle = getfield(obj, :handle)

    # Calculate field address: obj_ptr + offset
    # The getter expects the field address, not the object address
    field_addr = handle + field.offset

    # Create result Any for the getter
    result = Ref{LibTVMFFI.TVMFFIAny}()

    # Call the getter: int (*getter)(void* field_addr, TVMFFIAny* result)
    ret = ccall(field.getter, Cint, (Ptr{Cvoid}, Ptr{LibTVMFFI.TVMFFIAny}), field_addr, result)

    if ret != 0
        error("Failed to get field '$(field.name)'")
    end

    # Convert result to Julia value
    return copy_value(TVMAnyView(result[]))
end

"""
    set_field_value!(obj, field::FieldInfo, value) -> Nothing

Write a field value to an object using the reflection setter.
"""
function set_field_value!(obj, field::FieldInfo, value)
    if field.setter == C_NULL
        error("Field '$(field.name)' has no setter")
    end

    if !field.is_writable
        error("Field '$(field.name)' is read-only")
    end

    # Get pointer to the object
    handle = getfield(obj, :handle)

    # Calculate field address: obj_ptr + offset
    field_addr = handle + field.offset

    # Convert Julia value to TVMAny
    value_any = TVMAny(value)

    # Call the setter: int (*setter)(void* field_addr, const TVMFFIAny* value)
    # Use GC.@preserve to ensure value_any stays alive during the ccall
    GC.@preserve value_any begin
        ret = ccall(field.setter, Cint, (Ptr{Cvoid}, Ref{LibTVMFFI.TVMFFIAny}), field_addr, value_any.data)
    end

    if ret != 0
        error("Failed to set field '$(field.name)'")
    end

    return nothing
end

"""
    call_method(obj, method::MethodInfo, args...) -> Any

Call a method on an object.
For instance methods, obj is automatically prepended to the arguments.
"""
function call_method(obj, method::MethodInfo, args...)
    method_func = get_method_function(method)
    if method.is_static
        return method_func(args...)
    else
        # Instance method: prepend self
        handle = getfield(obj, :handle)
        tvm_obj = TVMObject(handle; borrowed=true)
        return method_func(tvm_obj, args...)
    end
end

#------------------------------------------------------------
# Section: Object Registration Macros
#------------------------------------------------------------

# Global cache for registered type indices (type -> index)
const _registered_type_indices = Dict{DataType, Int32}()

# Global cache for reflection info (type -> (fields, methods))
const _reflection_cache = Dict{DataType, Tuple{Vector{FieldInfo}, Vector{MethodInfo}}}()
const _reflection_cache_lock = ReentrantLock()

"""
    _get_reflection_cache(T::Type) -> Tuple{Vector{FieldInfo}, Vector{MethodInfo}}

Get cached reflection info for a type. Thread-safe with lazy initialization.
"""
function _get_reflection_cache(T::Type)
    lock(_reflection_cache_lock) do
        if haskey(_reflection_cache, T)
            return _reflection_cache[T]
        end

        # Try to get type info
        idx = get(_registered_type_indices, T, nothing)
        if idx === nothing
            # Type not registered, return empty
            result = (FieldInfo[], MethodInfo[])
            _reflection_cache[T] = result
            return result
        end

        type_info = get_type_info(idx)
        if type_info === nothing
            result = (FieldInfo[], MethodInfo[])
            _reflection_cache[T] = result
            return result
        end

        fields = get_fields(type_info)
        methods = get_methods(type_info)
        result = (fields, methods)
        _reflection_cache[T] = result
        return result
    end
end

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
- For types with `__ffi_init__`, use `ffi_init(T, args...)` or `T(args...)` to create instances

See also: [`register_object`](@ref), [`get_type_index`](@ref), [`type_index`](@ref)
"""
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
        # The actual mutable struct with handle and reflection cache
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

        # Reflection-based property access using global cache
        function Base.getproperty(obj::$(esc(type_name)), name::Symbol)
            # Handle the internal handle field
            if name === :handle
                return getfield(obj, :handle)
            end

            # Look up in global reflection cache
            fields, methods = _get_reflection_cache($(esc(type_name)))

            name_str = String(name)

            # Check fields first
            for field in fields
                if field.name == name_str
                    return get_field_value(obj, field)
                end
            end

            # Check methods
            for method in methods
                if method.name == name_str
                    # Return a bound method (closure that captures obj)
                    if method.is_static
                        method_func = get_method_function(method)
                        return (args...) -> method_func(args...)
                    else
                        return (args...) -> call_method(obj, method, args...)
                    end
                end
            end

            error("$($(string(type_name))) has no property '$name'")
        end

        function Base.setproperty!(obj::$(esc(type_name)), name::Symbol, value)
            if name === :handle
                error("Cannot modify handle field")
            end

            fields, _ = _get_reflection_cache($(esc(type_name)))
            name_str = String(name)

            for field in fields
                if field.name == name_str
                    set_field_value!(obj, field, value)
                    return value
                end
            end

            error("$($(string(type_name))) has no property '$name'")
        end

        function Base.propertynames(obj::$(esc(type_name)), private::Bool=false)
            names = Symbol[:handle]
            fields, methods = _get_reflection_cache($(esc(type_name)))

            for field in fields
                push!(names, Symbol(field.name))
            end
            for method in methods
                push!(names, Symbol(method.name))
            end

            return tuple(names...)
        end

        # Try to setup __ffi_init__ constructor
        _setup_ffi_init_constructor($(esc(type_name)), $(esc(type_key)))

        # External constructor for ffi_init (args...) syntax
        # This allows TypeName(arg1, arg2, ...) if __ffi_init__ exists
        function (::Type{$(esc(type_name))})(args...; kwargs...)
            if !has_ffi_init($(esc(type_name)))
                error("$($(string(type_name))) does not have a __ffi_init__ constructor. " *
                      "Use $($(string(type_name)))(handle; borrowed=...) instead.")
            end
            return ffi_init($(esc(type_name)), args...; kwargs...)
        end

        # Return the type
        $(esc(type_name))
    end
end

"""
    _setup_ffi_init_constructor(T::Type, type_key::String)

Setup a constructor that calls __ffi_init__ if available.
This enables `T(; field1=val1, field2=val2, ...)` syntax.
"""
function _setup_ffi_init_constructor(T::Type, type_key::String)
    # Look for __ffi_init__ method in reflection
    type_info = get_type_info(type_key)
    if type_info === nothing
        return  # No reflection info
    end

    methods = get_methods(type_info)
    init_method = nothing
    for m in methods
        if m.name == "__ffi_init__"
            init_method = m
            break
        end
    end

    if init_method === nothing
        return  # No __ffi_init__ method
    end

    # Store the init method for this type
    _ffi_init_methods[T] = init_method
end

# Cache for __ffi_init__ methods
const _ffi_init_methods = Dict{DataType, MethodInfo}()

"""
    has_ffi_init(T::Type) -> Bool

Check if a type has an __ffi_init__ method registered.
"""
has_ffi_init(T::Type) = haskey(_ffi_init_methods, T)

"""
    ffi_init(T::Type, args...; kwargs...) -> T

Create an instance of T using its __ffi_init__ method.

# Example
```julia
@register_object "testing.MyClass" struct MyClass end

# If MyClass has __ffi_init__(v_i64, v_f64), you can call:
obj = ffi_init(MyClass, 42, 3.14)
```
"""
function ffi_init(T::Type, args...; kwargs...)
    if !haskey(_ffi_init_methods, T)
        error("Type $T does not have an __ffi_init__ method")
    end

    init_method = _ffi_init_methods[T]
    method_func = get_method_function(init_method)

    # __ffi_init__ is typically static and returns an object handle
    result = method_func(args...; kwargs...)

    # If result is already the right type, return it
    if result isa T
        return result
    end

    # If result is a TVMObject, wrap it in T
    if result isa TVMObject
        return T(result.handle; borrowed=true)
    end

    # If result is a raw handle (shouldn't happen but just in case)
    if result isa Ptr{Cvoid} || result isa LibTVMFFI.TVMFFIObjectHandle
        return T(result; borrowed=false)
    end

    error("Unexpected result type from __ffi_init__: $(typeof(result))")
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
