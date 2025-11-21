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
        TVMObject(handle; borrowed=true)

    Create a TVMObject from a raw handle.

    # Arguments
    - `handle`: The raw object handle
    - `borrowed`: Reference semantics
      - `borrowed=true` (default): Borrowed reference, increment refcount (safe)
      - `borrowed=false`: Owned reference, take without IncRef (C gave us ownership)
    """
    function TVMObject(handle::LibTVMFFI.TVMFFIObjectHandle; borrowed::Bool = true)
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
