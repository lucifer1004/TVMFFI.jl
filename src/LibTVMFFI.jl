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
    LibTVMFFI

Low-level C API bindings for TVM FFI.

This module provides direct Julia bindings to the TVM FFI C API.
All functions follow the naming convention from c_api.h.

# Design Notes
- Uses ccall for direct C function invocation
- Matches C struct layouts exactly with Julia struct definitions
- No intermediate abstractions - keep it simple and direct
"""
module LibTVMFFI

using TVMFFI_jll: libtvm_ffi

# Section: Type definitions matching C API structures

# DLPack types (from dlpack.h, used by TVM)
@enum DLDeviceType::Int32 begin
    kDLCPU = 1
    kDLCUDA = 2
    kDLCUDAHost = 3
    kDLOpenCL = 4
    kDLVulkan = 7
    kDLMetal = 8
    kDLVPI = 9
    kDLROCM = 10
    kDLROCMHost = 11
    kDLExtDev = 12
    kDLCUDAManaged = 13
    kDLOneAPI = 14
    kDLWebGPU = 15
    kDLHexagon = 16
    kDLMAIA = 17
    kDLTrn = 18
end

"""
    DLDevice

Device context for array execution.

# Fields
- `device_type::Int32`: Device type (CPU, CUDA, etc.)
- `device_id::Int32`: Device ID (e.g., GPU 0, 1, 2...)
"""
struct DLDevice
    device_type::Int32
    device_id::Int32
end

@enum DLDataTypeCode::UInt8 begin
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLBfloat = 4
    kDLOpaqueHandle = 3
    kDLComplex = 5
    kDLBool = 6
    kDLFloat8_e3m4 = 7
    kDLFloat8_e4m3 = 8
    kDLFloat8_e4m3b11fnuz = 9
    kDLFloat8_e4m3fn = 10
    kDLFloat8_e4m3fnuz = 11
    kDLFloat8_e5m2 = 12
    kDLFloat8_e5m2fnuz = 13
    kDLFloat8_e8m0fnu = 14
    kDLFloat6_e2m3fn = 15
    kDLFloat6_e3m2fn = 16
    kDLFloat4_e2m1fn = 17
end

"""
    DLDataType

Data type descriptor.

# Fields
- `code::UInt8`: Type code (int, uint, float, etc.)
- `bits::UInt8`: Number of bits
- `lanes::UInt16`: Number of lanes for vector types (1 for scalar)
"""
struct DLDataType
    code::UInt8
    bits::UInt8
    lanes::UInt16
end

# TVM FFI Type Index
# Note: In C, kTVMFFIStaticObjectBegin and kTVMFFIObject both have value 64.
# Julia @enum requires unique values, so we define boundary markers as constants.
@enum TVMFFITypeIndex::Int32 begin
    kTVMFFIAny = -1
    kTVMFFINone = 0
    kTVMFFIInt = 1
    kTVMFFIBool = 2
    kTVMFFIFloat = 3
    kTVMFFIOpaquePtr = 4
    kTVMFFIDataType = 5
    kTVMFFIDevice = 6
    kTVMFFIDLTensorPtr = 7
    kTVMFFIRawStr = 8
    kTVMFFIByteArrayPtr = 9
    kTVMFFIObjectRValueRef = 10
    kTVMFFISmallStr = 11
    kTVMFFISmallBytes = 12
    # Static object types start at 64
    kTVMFFIObject = 64
    kTVMFFIStr = 65
    kTVMFFIBytes = 66
    kTVMFFIError = 67
    kTVMFFIFunction = 68
    kTVMFFIShape = 69
    kTVMFFITensor = 70
    kTVMFFIArray = 71
    kTVMFFIMap = 72
    kTVMFFIModule = 73
    kTVMFFIOpaquePyObject = 74
end

# Boundary markers (not enum members due to value conflicts)
const kTVMFFIStaticObjectBegin = Int32(64)   # Same as kTVMFFIObject
const kTVMFFIStaticObjectEnd = Int32(75)     # After last static type
const kTVMFFIDynObjectBegin = Int32(128)     # Dynamic types start here

# Object handle - opaque pointer
const TVMFFIObjectHandle = Ptr{Cvoid}

"""
    TVMFFIVersion

TVM FFI version information.
"""
struct TVMFFIVersion
    major::UInt32
    minor::UInt32
    patch::UInt32
end

"""
    TVMFFIObject

Object header for all heap-allocated TVM FFI objects.
Contains reference counting and type information.
"""
struct TVMFFIObject
    combined_ref_count::UInt64  # Strong ref in lower 32 bits, weak ref in upper 32 bits
    type_index::Int32
    __padding::UInt32
    deleter::Ptr{Cvoid}  # Function pointer or alignment padding
end

"""
    TVMFFIAny

Type-erased value container. Can hold POD types or object references.
"""
struct TVMFFIAny
    type_index::Int32
    small_str_len::UInt32  # Only used for small strings/bytes
    # Union of 8 bytes - Julia doesn't have unions, so we use a single UInt64
    # and provide accessors
    data::UInt64
end

"""
    TVMFFIByteArray

Byte array descriptor used by String and Bytes objects.
"""
struct TVMFFIByteArray
    data::Ptr{UInt8}
    size::UInt
end

"""
    TVMFFIShapeCell

Shape cell used in shape object following header.
"""
struct TVMFFIShapeCell
    data::Ptr{Int64}
    size::UInt
end

"""
    TVMFFIErrorCell

Error cell structure following the object header in error objects.
"""
struct TVMFFIErrorCell
    kind::TVMFFIByteArray
    message::TVMFFIByteArray
    backtrace::TVMFFIByteArray
    update_backtrace::Ptr{Cvoid}  # Function pointer
end

"""
    TVMFFIFunctionCell

Function cell structure following the object header in function objects.
"""
struct TVMFFIFunctionCell
    safe_call::Ptr{Cvoid}  # TVMFFISafeCallType
    cpp_call::Ptr{Cvoid}   # Nullable C++ call pointer
end

"""
    TVMFFIOpaqueObjectCell

Opaque object cell for storing external handles (e.g., Python PyObject*).
"""
struct TVMFFIOpaqueObjectCell
    handle::Ptr{Cvoid}
end

"""
    TVMFFITypeInfo

Runtime type information for object type checking.
Matches the C struct layout from c_api.h.
"""
struct TVMFFITypeInfo
    type_index::Int32
    type_depth::Int32
    type_key::TVMFFIByteArray
    type_ancestors::Ptr{Cvoid}  # const struct TVMFFITypeInfo**
    type_key_hash::UInt64
    num_fields::Int32
    num_methods::Int32
    fields::Ptr{Cvoid}  # const TVMFFIFieldInfo*
    methods::Ptr{Cvoid}  # const TVMFFIMethodInfo*
    metadata::Ptr{Cvoid}  # const TVMFFITypeMetadata*
end

# Section: Core C API functions
# Organized to match c_api.h structure

#------------------------------------------------------------
# Section: Version API
#------------------------------------------------------------

"""
    TVMFFIGetVersion() -> TVMFFIVersion

Get the TVM FFI version from the current C ABI.
"""
function TVMFFIGetVersion()
    version = Ref{TVMFFIVersion}()
    @ccall libtvm_ffi.TVMFFIGetVersion(version::Ptr{TVMFFIVersion})::Cvoid
    return version[]
end

#------------------------------------------------------------
# Section: Basic object API
#------------------------------------------------------------

"""
    TVMFFIObjectIncRef(obj::TVMFFIObjectHandle)

Increase the strong reference count of an object.
"""
function TVMFFIObjectIncRef(obj::TVMFFIObjectHandle)
    @ccall libtvm_ffi.TVMFFIObjectIncRef(obj::TVMFFIObjectHandle)::Cint
end

"""
    TVMFFIObjectDecRef(obj::TVMFFIObjectHandle)

Decrease the strong reference count of an object.
"""
function TVMFFIObjectDecRef(obj::TVMFFIObjectHandle)
    @ccall libtvm_ffi.TVMFFIObjectDecRef(obj::TVMFFIObjectHandle)::Cint
end

"""
    TVMFFIObjectCreateOpaque(handle, type_index, deleter) -> (Int32, TVMFFIObjectHandle)

Create an opaque object by passing in handle, type_index and deleter.
Useful for wrapping external objects (e.g., Python PyObject*).
"""
function TVMFFIObjectCreateOpaque(
        handle::Ptr{Cvoid},
        type_index::Int32,
        deleter::Ptr{Cvoid}
)
    out_handle = Ref{TVMFFIObjectHandle}(C_NULL)
    ret = @ccall libtvm_ffi.TVMFFIObjectCreateOpaque(
        handle::Ptr{Cvoid},
        type_index::Int32,
        deleter::Ptr{Cvoid},
        out_handle::Ptr{TVMFFIObjectHandle}
    )::Cint
    return ret, out_handle[]
end

"""
    TVMFFITypeKeyToIndex(key) -> (Int32, Int32)

Convert type key to type index.
"""
function TVMFFITypeKeyToIndex(key::TVMFFIByteArray)
    out_index = Ref{Int32}(0)
    ret = @ccall libtvm_ffi.TVMFFITypeKeyToIndex(
        key::Ref{TVMFFIByteArray}, out_index::Ptr{Int32})::Cint
    return ret, out_index[]
end

#-----------------------------------------------------------------------
# Section: Basic function calling API for function implementation
#-----------------------------------------------------------------------

"""
    TVMFFIFunctionCreate(resource_handle, safe_call, deleter) -> (Int32, TVMFFIObjectHandle)

Create a FFI function from C callbacks.
"""
function TVMFFIFunctionCreate(
        resource_handle::Ptr{Cvoid},
        safe_call::Ptr{Cvoid},
        deleter::Ptr{Cvoid}
)
    out_handle = Ref{TVMFFIObjectHandle}(C_NULL)
    ret = @ccall libtvm_ffi.TVMFFIFunctionCreate(
        resource_handle::Ptr{Cvoid},
        safe_call::Ptr{Cvoid},
        deleter::Ptr{Cvoid},
        out_handle::Ptr{TVMFFIObjectHandle}
    )::Cint
    return ret, out_handle[]
end

"""
    TVMFFIFunctionGetGlobal(name) -> (Int32, TVMFFIObjectHandle)

Get a global function by name.
"""
function TVMFFIFunctionGetGlobal(name::TVMFFIByteArray)
    out_handle = Ref{TVMFFIObjectHandle}(C_NULL)
    ret = ccall((:TVMFFIFunctionGetGlobal, libtvm_ffi), Cint,
                (Ref{TVMFFIByteArray}, Ptr{TVMFFIObjectHandle}),
                Ref(name), out_handle)
    return ret, out_handle[]
end

"""
    TVMFFIAnyViewToOwnedAny(any_view) -> (Int32, TVMFFIAny)

Convert an AnyView to an owned Any.
"""
function TVMFFIAnyViewToOwnedAny(any_view::TVMFFIAny)
    out_any = Ref{TVMFFIAny}(TVMFFIAny(Int32(kTVMFFINone), 0, 0))
    ret = @ccall libtvm_ffi.TVMFFIAnyViewToOwnedAny(
        any_view::Ref{TVMFFIAny}, out_any::Ptr{TVMFFIAny})::Cint
    return ret, out_any[]
end

"""
    TVMFFIFunctionCall(func, args, num_args, result)

Call a TVM function with arguments.
"""
function TVMFFIFunctionCall(
        func::TVMFFIObjectHandle,
        args::Ptr{TVMFFIAny},
        num_args::Int32,
        result::Ptr{TVMFFIAny}
)
    @ccall libtvm_ffi.TVMFFIFunctionCall(func::TVMFFIObjectHandle, args::Ptr{TVMFFIAny},
        num_args::Int32, result::Ptr{TVMFFIAny})::Cint
end

"""
    TVMFFIErrorMoveFromRaised() -> TVMFFIObjectHandle

Move the last error from TLS (Thread Local Storage).
"""
function TVMFFIErrorMoveFromRaised()
    out_handle = Ref{TVMFFIObjectHandle}(C_NULL)
    @ccall libtvm_ffi.TVMFFIErrorMoveFromRaised(out_handle::Ptr{TVMFFIObjectHandle})::Cvoid
    return out_handle[]
end

"""
    TVMFFIErrorSetRaised(error::TVMFFIObjectHandle)

Set a raised error in TLS.
"""
function TVMFFIErrorSetRaised(error::TVMFFIObjectHandle)
    @ccall libtvm_ffi.TVMFFIErrorSetRaised(error::TVMFFIObjectHandle)::Cvoid
end

"""
    TVMFFIErrorSetRaisedFromCStr(kind, message)

Set a raised error in TLS from C strings (convenient method).
"""
function TVMFFIErrorSetRaisedFromCStr(kind::Cstring, message::Cstring)
    @ccall libtvm_ffi.TVMFFIErrorSetRaisedFromCStr(kind::Cstring, message::Cstring)::Cvoid
end

"""
    TVMFFIErrorCreate(kind, message, backtrace) -> (Int32, TVMFFIObjectHandle)

Create an error object.
Returns (return_code, error_handle).
"""
function TVMFFIErrorCreate(
        kind::TVMFFIByteArray,
        message::TVMFFIByteArray,
        backtrace::TVMFFIByteArray
)
    out_handle = Ref{TVMFFIObjectHandle}(C_NULL)
    ret = @ccall libtvm_ffi.TVMFFIErrorCreate(
        kind::Ref{TVMFFIByteArray},
        message::Ref{TVMFFIByteArray},
        backtrace::Ref{TVMFFIByteArray},
        out_handle::Ptr{TVMFFIObjectHandle}
    )::Cint
    return ret, out_handle[]
end

#------------------------------------------------------------
# Section: string/bytes support APIs
#------------------------------------------------------------

"""
    TVMFFITypeGetOrAllocIndex(key, static_type_index, type_depth, num_child_slots, 
                              child_slots_can_overflow, parent_type_index) -> Int32

Get type index from type key, allocating a new one if not found.
Returns the type index directly (not an error code).

# Arguments
- `key`: The type key byte array
- `static_type_index`: Static type index (use -1 for dynamic types)  
- `type_depth`: Depth in type hierarchy
- `num_child_slots`: Number of child type slots
- `child_slots_can_overflow`: Whether child slots can overflow (0 or 1)
- `parent_type_index`: Parent type index (use -1 for no parent)
"""
function TVMFFITypeGetOrAllocIndex(
    key::TVMFFIByteArray,
    static_type_index::Int32 = Int32(-1),
    type_depth::Int32 = Int32(0),
    num_child_slots::Int32 = Int32(0),
    child_slots_can_overflow::Int32 = Int32(0),
    parent_type_index::Int32 = Int32(-1)
)
    idx = @ccall libtvm_ffi.TVMFFITypeGetOrAllocIndex(
        key::Ref{TVMFFIByteArray}, 
        static_type_index::Int32,
        type_depth::Int32,
        num_child_slots::Int32,
        child_slots_can_overflow::Int32,
        parent_type_index::Int32
    )::Int32
    return idx
end

"""
    TVMFFIGetTypeInfo(type_index::Int32) -> Ptr{TVMFFITypeInfo}

Get type information by type index.
Returns a pointer to the type info structure (owned by runtime, do not free).
"""
function TVMFFIGetTypeInfo(type_index::Int32)
    ptr = @ccall libtvm_ffi.TVMFFIGetTypeInfo(type_index::Int32)::Ptr{TVMFFITypeInfo}
    return ptr
end

"""
    TVMFFIDataTypeFromString(str) -> (Int32, DLDataType)

Parse a string to DLDataType.
"""
function TVMFFIDataTypeFromString(str::TVMFFIByteArray)
    out_dtype = Ref{DLDataType}()
    ret = @ccall libtvm_ffi.TVMFFIDataTypeFromString(
        str::Ref{TVMFFIByteArray}, out_dtype::Ptr{DLDataType})::Cint
    return ret, out_dtype[]
end

"""
    TVMFFIDataTypeToString(dtype) -> (Int32, TVMFFIAny)

Convert DLDataType to string.
"""
function TVMFFIDataTypeToString(dtype::DLDataType)
    out_any = Ref{TVMFFIAny}(TVMFFIAny(Int32(kTVMFFINone), 0, 0))
    ret = @ccall libtvm_ffi.TVMFFIDataTypeToString(
        dtype::Ref{DLDataType}, out_any::Ptr{TVMFFIAny})::Cint
    return ret, out_any[]
end

"""
    TVMFFIStringFromByteArray(input) -> (Int32, TVMFFIAny)

Create a String from TVMFFIByteArray.
"""
function TVMFFIStringFromByteArray(input::TVMFFIByteArray)
    out_any = Ref{TVMFFIAny}(TVMFFIAny(Int32(kTVMFFINone), 0, 0))
    ret = @ccall libtvm_ffi.TVMFFIStringFromByteArray(
        input::Ref{TVMFFIByteArray}, out_any::Ptr{TVMFFIAny})::Cint
    return ret, out_any[]
end

"""
    TVMFFIBytesFromByteArray(input) -> (Int32, TVMFFIAny)

Create a Bytes from TVMFFIByteArray.
"""
function TVMFFIBytesFromByteArray(input::TVMFFIByteArray)
    out_any = Ref{TVMFFIAny}(TVMFFIAny(Int32(kTVMFFINone), 0, 0))
    ret = @ccall libtvm_ffi.TVMFFIBytesFromByteArray(
        input::Ref{TVMFFIByteArray}, out_any::Ptr{TVMFFIAny})::Cint
    return ret, out_any[]
end

#-----------------------------------------------------------------------
# Section: Backend noexcept functions for internal use
#-----------------------------------------------------------------------

"""
    TVMFFIFunctionSetGlobal(name, func, override) -> Int32

Register a global function to runtime's global table.
"""
function TVMFFIFunctionSetGlobal(
        name::TVMFFIByteArray,
        func::TVMFFIObjectHandle,
        override::Int32
)
    @ccall libtvm_ffi.TVMFFIFunctionSetGlobal(
        name::Ref{TVMFFIByteArray},
        func::TVMFFIObjectHandle,
        override::Cint
    )::Cint
end

#---------------------------------------------------------------
# Section: Inline accessor functions (from C++ inline functions)
#---------------------------------------------------------------

"""
    TVMFFIObjectGetTypeIndex(obj::TVMFFIObjectHandle) -> Int32

Get the type index of an object.
"""
function TVMFFIObjectGetTypeIndex(obj::TVMFFIObjectHandle)
    unsafe_load(Ptr{TVMFFIObject}(obj)).type_index
end

"""
    TVMFFIErrorGetCellPtr(obj::TVMFFIObjectHandle) -> Ptr{TVMFFIErrorCell}

Get pointer to the error cell from an error object.
"""
function TVMFFIErrorGetCellPtr(obj::TVMFFIObjectHandle)
    Ptr{TVMFFIErrorCell}(obj + sizeof(TVMFFIObject))
end

"""
    TVMFFIBytesGetByteArrayPtr(obj::TVMFFIObjectHandle) -> Ptr{TVMFFIByteArray}

Get pointer to the byte array from a string or bytes object.
"""
function TVMFFIBytesGetByteArrayPtr(obj::TVMFFIObjectHandle)
    Ptr{TVMFFIByteArray}(obj + sizeof(TVMFFIObject))
end

"""
    TVMFFIFunctionGetCellPtr(obj::TVMFFIObjectHandle) -> Ptr{TVMFFIFunctionCell}

Get pointer to the function cell from a function object.
"""
function TVMFFIFunctionGetCellPtr(obj::TVMFFIObjectHandle)
    Ptr{TVMFFIFunctionCell}(obj + sizeof(TVMFFIObject))
end

"""
    TVMFFIOpaqueObjectGetCellPtr(obj::TVMFFIObjectHandle) -> Ptr{TVMFFIOpaqueObjectCell}

Get pointer to the opaque object cell from an opaque object.
"""
function TVMFFIOpaqueObjectGetCellPtr(obj::TVMFFIObjectHandle)
    Ptr{TVMFFIOpaqueObjectCell}(obj + sizeof(TVMFFIObject))
end

"""
    TVMFFIShapeGetCellPtr(obj::TVMFFIObjectHandle) -> Ptr{TVMFFIShapeCell}

Get pointer to the shape cell from a shape object.
"""
function TVMFFIShapeGetCellPtr(obj::TVMFFIObjectHandle)
    Ptr{TVMFFIShapeCell}(obj + sizeof(TVMFFIObject))
end

"""
    TVMFFISmallBytesGetContentByteArray(value::TVMFFIAny) -> TVMFFIByteArray

Get the content of a small string/bytes in byte array format.
"""
function TVMFFISmallBytesGetContentByteArray(value::TVMFFIAny)
    # Extract the v_bytes field and length from TVMFFIAny
    # v_bytes is the 8-byte data field, small_str_len indicates the length
    data_ptr = Base.unsafe_convert(Ptr{UInt8}, Ref(value, 3))  # Offset to data field
    TVMFFIByteArray(data_ptr, value.small_str_len)
end

"""
    TVMFFIDLDeviceFromIntPair(device_type::Int32, device_id::Int32) -> DLDevice

Create a DLDevice from device type and device ID.
"""
function TVMFFIDLDeviceFromIntPair(device_type::Int32, device_id::Int32)
    DLDevice(device_type, device_id)
end

end # module LibTVMFFI
