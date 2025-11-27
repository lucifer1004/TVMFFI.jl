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

"""
    DLTensor

DLPack tensor structure (from dlpack.h).
This is a C-compatible struct representing a multi-dimensional array.

# Note on GPU Pointers
For GPU arrays (CuArray, ROCArray, etc.), the data field contains a GPU device
pointer, not a CPU pointer. We use UInt64 to store the pointer value and
reinterpret it as needed, since GPU pointers can't be directly converted to Ptr{Cvoid}.
"""
struct DLTensor
    data::Ptr{Cvoid}
    device::DLDevice
    ndim::Int32
    dtype::DLDataType
    shape::Ptr{Int64}
    strides::Ptr{Int64}
    byte_offset::UInt64
end

"""
    DLTensor(data_ptr, device, ndim, dtype, shape, strides, byte_offset)

Construct DLTensor with automatic GPU pointer handling.
"""
function DLTensor(
        data_ptr,
        device::DLDevice,
        ndim::Int32,
        dtype::DLDataType,
        shape::Ptr{Int64},
        strides::Ptr{Int64},
        byte_offset::UInt64
)
    # Convert GPU pointers (CuPtr, MtlPtr, etc.) to generic pointer
    # Handle different pointer types:
    # - Ptr: Standard Julia pointer, convert directly
    # - isbits GPU pointers (CuPtr, ROCPtr): Use reinterpret
    # - Non-isbits GPU pointers (MtlPtr): Extract via Metal.contents + offset
    ptr_as_uint = if data_ptr isa Ptr
        UInt(data_ptr)
    elseif isbitstype(typeof(data_ptr))
        # GPU pointer that is isbits (CuPtr, ROCPtr, etc.)
        # Get the raw pointer value via reinterpret
        reinterpret(UInt, data_ptr)
    else
        # Non-isbits GPU pointer (e.g., Metal.MtlPtr)
        # Check if it's Metal.MtlPtr by checking field names
        ptr_type = typeof(data_ptr)
        if hasfield(ptr_type, :buffer) && hasfield(ptr_type, :offset)
            # Metal.jl specific: MtlPtr{T} has buffer and offset fields
            # For Metal, we need to pass the MTLBuffer object pointer, not the data pointer
            # because Metal needs the buffer object to access metadata (size, storage mode, etc.)
            buffer = getfield(data_ptr, :buffer)
            # Extract MTLBuffer object pointer from buffer.ptr
            mtl_buffer_obj = getfield(buffer, :ptr)
            # Convert ObjectiveC.id to pointer value
            # This will be reinterpreted as id<MTLBuffer> in C++ code
            UInt(reinterpret(Ptr{Cvoid}, mtl_buffer_obj))
        else
            error("Unsupported GPU pointer type: $(ptr_type). " *
                  "Expected Ptr, isbits GPU pointer, or Metal.MtlPtr")
        end
    end

    # Convert to Ptr{Cvoid}
    data_cvoid = reinterpret(Ptr{Cvoid}, ptr_as_uint)

    return DLTensor(data_cvoid, device, ndim, dtype, shape, strides, byte_offset)
end

"""
    TVMTensor

TVM tensor wrapper with automatic memory management.

Provides accessors for shape, dtype, and device information.
"""
mutable struct TVMTensor
    handle::LibTVMFFI.TVMFFIObjectHandle

    """
        TVMTensor(handle; borrowed)

    Create a TVMTensor from a raw handle. Internal API - users should not call directly.

    # Arguments
    - `handle`: The raw tensor handle
    - `borrowed`: Reference semantics (REQUIRED - no default to prevent misuse)
      - `borrowed=true`: Borrowed reference, increment refcount
      - `borrowed=false`: Owned reference, take without IncRef (C gave us ownership)
    """
    function TVMTensor(handle::LibTVMFFI.TVMFFIObjectHandle; borrowed::Bool)
        if handle == C_NULL
            error("Cannot create TVMTensor from NULL handle")
        end

        # Copy reference if borrowed
        if borrowed
            LibTVMFFI.TVMFFIObjectIncRef(handle)
        end

        tensor = new(handle)

        # Finalizer
        finalizer(tensor) do t
            if t.handle != C_NULL
                LibTVMFFI.TVMFFIObjectDecRef(t.handle)
            end
        end

        return tensor
    end
end

"""
    get_dltensor_ptr(tensor::TVMTensor) -> Ptr{DLTensor}

Get pointer to the underlying DLTensor structure.
"""
function get_dltensor_ptr(tensor::TVMTensor)
    # DLTensor follows immediately after TVMFFIObject header
    Ptr{DLTensor}(tensor.handle + sizeof(LibTVMFFI.TVMFFIObject))
end

"""
    shape(tensor::TVMTensor) -> Vector{Int64}

Get the shape of the tensor as a vector.

# Note
Julia arrays typically use `size()` which returns a tuple.
This function returns a vector for compatibility with some use cases.
"""
function shape(tensor::TVMTensor)
    dltensor_ptr = get_dltensor_ptr(tensor)
    dltensor = unsafe_load(dltensor_ptr)

    ndim = Int(dltensor.ndim)
    if ndim == 0
        return Int64[]
    end

    shape_vec = unsafe_wrap(Array, dltensor.shape, ndim)
    return copy(shape_vec)
end

"""
    Base.size(tensor::TVMTensor) -> Tuple

Get the size of the tensor as a tuple (Julia standard).
"""
Base.size(tensor::TVMTensor) = Tuple(shape(tensor))

"""
    Base.size(tensor::TVMTensor, dim::Int) -> Int

Get the size of a specific dimension.
"""
function Base.size(tensor::TVMTensor, dim::Int)
    s = size(tensor)
    if dim < 1 || dim > length(s)
        return 1  # Julia convention for out-of-bounds dimensions
    end
    return s[dim]
end

"""
    Base.ndims(tensor::TVMTensor) -> Int

Get the number of dimensions.
"""
function Base.ndims(tensor::TVMTensor)
    dltensor_ptr = get_dltensor_ptr(tensor)
    dltensor = unsafe_load(dltensor_ptr)
    return Int(dltensor.ndim)
end

"""
    Base.length(tensor::TVMTensor) -> Int

Get the total number of elements.
"""
function Base.length(tensor::TVMTensor)
    prod(size(tensor))
end

"""
    dtype(tensor::TVMTensor) -> DLDataType

Get the data type of the tensor.
"""
function dtype(tensor::TVMTensor)
    dltensor_ptr = get_dltensor_ptr(tensor)
    dltensor = unsafe_load(dltensor_ptr)
    return dltensor.dtype
end

"""
    device(tensor::TVMTensor) -> DLDevice

Get the device of the tensor.
"""
function device(tensor::TVMTensor)
    dltensor_ptr = get_dltensor_ptr(tensor)
    dltensor = unsafe_load(dltensor_ptr)
    return dltensor.device
end

"""
    strides(tensor::TVMTensor) -> Vector{Int64}

Get the strides of the tensor.
"""
function strides(tensor::TVMTensor)
    dltensor_ptr = get_dltensor_ptr(tensor)
    dltensor = unsafe_load(dltensor_ptr)

    if dltensor.strides == C_NULL
        # Compute default C-contiguous strides
        shape_vec = shape(tensor)
        ndim = length(shape_vec)
        strides_vec = ones(Int64, ndim)

        for i in (ndim - 1):-1:1
            strides_vec[i] = strides_vec[i + 1] * shape_vec[i + 1]
        end

        return strides_vec
    else
        ndim = Int(dltensor.ndim)
        strides_vec = unsafe_wrap(Array, dltensor.strides, ndim)
        return copy(strides_vec)
    end
end

"""
    is_contiguous(tensor::TVMTensor) -> Bool

Check if the tensor is contiguous in memory.
"""
function is_contiguous(tensor::TVMTensor)
    shape_vec = shape(tensor)
    strides_vec = strides(tensor)
    ndim = length(shape_vec)

    expected_stride = 1
    for i in ndim:-1:1
        if strides_vec[i] != expected_stride
            return false
        end
        expected_stride *= shape_vec[i]
    end

    return true
end

"""
    data_ptr(tensor::TVMTensor) -> Ptr{Cvoid}

Get the raw data pointer of the tensor.
"""
function data_ptr(tensor::TVMTensor)
    dltensor_ptr = get_dltensor_ptr(tensor)
    dltensor = unsafe_load(dltensor_ptr)
    return dltensor.data
end

# Pretty printing
function Base.show(io::IO, tensor::TVMTensor)
    shape_tuple = size(tensor)
    dt = dtype(tensor)
    dev = device(tensor)

    print(io, "TVMTensor{", string(dt), "}(")
    print(io, "shape=", shape_tuple, ", ")
    print(io, "device=", dev, ")")
end

"""
    Base.summary(tensor::TVMTensor) -> String

Get a summary string for the tensor.
"""
function Base.summary(io::IO, tensor::TVMTensor)
    shape_tuple = size(tensor)
    dt = dtype(tensor)
    print(io, join(shape_tuple, "×"), " TVMTensor{", string(dt), "}")
end

"""
    TensorOwnership

Enum indicating who manages the underlying tensor data.

- `JuliaOwned`: Julia GC manages the data (source is Array/CuArray)
- `TVMOwned`: TVM refcount manages the data (no source, used internally)
"""
@enum TensorOwnership::Int8 begin
    JuliaOwned = 0
    TVMOwned = 1
end

"""
    TensorView{T, S}

Lightweight DLTensor wrapper for Julia ↔ TVM tensor exchange.

Used for:
- Julia → TVM: Wrap Julia array as DLTensor for passing to TVM functions
- Internal: Temporary view during `to_dlmanaged_tensor`

# Type Parameters
- `T`: Element type (Float32, Int64, etc.)
- `S`: Source type (Array, CuArray, or Nothing)

# Fields
- `dltensor::DLTensor`: DLPack tensor structure
- `shape::Vector{Int64}`: Shape array (kept alive for dltensor.shape pointer)
- `strides::Vector{Int64}`: Strides array (kept alive for dltensor.strides pointer)
- `source::S`: Source array (kept alive to prevent GC)
- `ownership::TensorOwnership`: Who manages the underlying data

# Example
```julia
arr = rand(Float32, 3, 4)
view = TensorView(arr)  # JuliaOwned, source = arr
```
"""
mutable struct TensorView{T, S}
    dltensor::DLTensor
    shape::Vector{Int64}
    strides::Vector{Int64}
    source::S
    ownership::TensorOwnership

    function TensorView{T, S}(
            dltensor::DLTensor,
            shape::Vector{Int64},
            strides::Vector{Int64},
            source::S,
            ownership::TensorOwnership
    ) where {T, S}
        new{T, S}(dltensor, shape, strides, source, ownership)
    end
end

# Outer constructor for CPU arrays (JuliaOwned)
function TensorView(
        arr::Union{Array{T}, SubArray{T}},
        device::DLDevice = cpu()
) where {T}
    # Get shape
    shape_tuple = size(arr)
    ndim = length(shape_tuple)
    shape_vec = collect(Int64, shape_tuple)

    # Get strides - Julia provides this for both Array and SubArray
    arr_strides = Base.strides(arr)
    strides_vec = collect(Int64, arr_strides)

    # Get data pointer - Julia's pointer() handles SubArray correctly
    data_ptr = pointer(arr)

    # byte_offset is 0 (pointer already points to first element)
    byte_offset = UInt64(0)

    # Get dtype
    dt = DLDataType(T)

    # Create DLTensor
    dltensor = DLTensor(
        data_ptr,
        device,
        Int32(ndim),
        dt,
        pointer(shape_vec),
        pointer(strides_vec),
        byte_offset
    )

    return TensorView{T, typeof(arr)}(dltensor, shape_vec, strides_vec, arr, JuliaOwned)
end

"""
    Base.Ref(view::TensorView) -> Ref{DLTensor}

Get a reference to the underlying DLTensor for passing to C functions.
The TensorView keeps all data alive.
"""
Base.Ref(view::TensorView) = Ref(view.dltensor)

"""
    Base.unsafe_convert(::Type{Ptr{DLTensor}}, view::TensorView)

Convert TensorView to pointer for C calls.
This is called automatically by ccall.
"""
function Base.unsafe_convert(::Type{Ptr{DLTensor}}, view::TensorView)
    # Get pointer to the dltensor field within the TensorView
    # The dltensor is the first field, so we can get its address
    return Ptr{DLTensor}(pointer_from_objref(view))
end

# Type system integration for TVMTensor
type_index(tensor::TVMTensor) = LibTVMFFI.TVMFFIObjectGetTypeIndex(tensor.handle)
type_index(::Type{TVMTensor}) = Int32(LibTVMFFI.kTVMFFITensor)
type_key(::Type{TVMTensor}) = "ffi.Tensor"

#=============================================================================
  DLManagedTensor - DLPack Protocol Structure
=============================================================================#

"""
    DLManagedTensor

DLPack protocol structure for tensor exchange between libraries.
This is a C-compatible struct matching the DLPack specification.

# Fields
- `dl_tensor::DLTensor`: The tensor data
- `manager_ctx::Ptr{Cvoid}`: Context for the deleter callback
- `deleter::Ptr{Cvoid}`: Function pointer called when tensor is no longer needed
"""
mutable struct DLManagedTensor
    dl_tensor::DLTensor
    manager_ctx::Ptr{Cvoid}
    deleter::Ptr{Cvoid}
end

#=============================================================================
  TensorView from TVMTensor (TVMOwned)
=============================================================================#

"""
    TensorView(tensor::TVMTensor) -> TensorView

Create a TensorView from a TVMTensor (zero-copy).

The TensorView holds a reference to the TVMTensor, keeping it alive.
The underlying data is managed by TVM's reference counting.

# Example
```julia
tvm_tensor = some_tvm_function()
view = TensorView(tvm_tensor)
# view.source === tvm_tensor (keeps it alive)
```
"""
function TensorView(tensor::TVMTensor)
    # Get DLTensor from TVMTensor
    dltensor_ptr = get_dltensor_ptr(tensor)
    dltensor = unsafe_load(dltensor_ptr)

    # Extract type from dtype
    T = dtype_to_julia_type(dltensor.dtype)

    # Copy shape and strides (we need our own arrays for the pointers)
    ndim = Int(dltensor.ndim)
    shape_vec = if ndim > 0
        unsafe_wrap(Array, dltensor.shape, ndim) |> copy
    else
        Int64[]
    end

    strides_vec = if dltensor.strides != C_NULL && ndim > 0
        unsafe_wrap(Array, dltensor.strides, ndim) |> copy
    else
        # Compute default C-contiguous strides
        _compute_contiguous_strides(shape_vec)
    end

    # Create new DLTensor with our shape/strides pointers
    new_dltensor = DLTensor(
        dltensor.data,
        dltensor.device,
        dltensor.ndim,
        dltensor.dtype,
        pointer(shape_vec),
        pointer(strides_vec),
        dltensor.byte_offset
    )

    return TensorView{T, TVMTensor}(new_dltensor, shape_vec, strides_vec, tensor, TVMOwned)
end

#=============================================================================
  Helper Functions
=============================================================================#

"""
Compute C-contiguous (row-major) strides from shape.
"""
function _compute_c_contiguous_strides(shape::Vector{Int64})
    ndim = length(shape)
    if ndim == 0
        return Int64[]
    end
    strides = ones(Int64, ndim)
    for i in (ndim - 1):-1:1
        strides[i] = strides[i + 1] * shape[i + 1]
    end
    return strides
end

"""
Compute F-contiguous (column-major, Julia-style) strides from shape.
"""
function _compute_f_contiguous_strides(shape::Vector{Int64})
    ndim = length(shape)
    if ndim == 0
        return Int64[]
    end
    strides = ones(Int64, ndim)
    for i in 2:ndim
        strides[i] = strides[i - 1] * shape[i - 1]
    end
    return strides
end

# Alias for backward compatibility
_compute_contiguous_strides(shape) = _compute_c_contiguous_strides(shape)

"""
    to_dlmanaged_tensor(view::TensorView) -> Ptr{DLManagedTensor}

Create a DLManagedTensor from a TensorView for passing to external libraries.

The caller is responsible for ensuring the TensorView stays alive while
the DLManagedTensor is in use.

# Returns
Pointer to a DLManagedTensor (allocated in DLMANAGED_POOL)
"""
const _DLMANAGED_POOL = Dict{Ptr{Cvoid}, Any}()
const _DLMANAGED_POOL_LOCK = ReentrantLock()

# Flag to prevent operations during Julia shutdown
# Shared across tensor.jl, dlpack.jl, and function.jl
const _julia_is_exiting = Ref(false)

function _dlmanaged_deleter(manager_ctx::Ptr{Cvoid})
    # Skip if Julia is exiting - runtime may be in inconsistent state
    _julia_is_exiting[] && return nothing

    lock(_DLMANAGED_POOL_LOCK) do
        delete!(_DLMANAGED_POOL, manager_ctx)
    end
    return nothing
end

const _DLMANAGED_DELETER_CPTR = Ref{Ptr{Cvoid}}(C_NULL)

function _init_dlmanaged_deleter()
    _DLMANAGED_DELETER_CPTR[] = @cfunction(_dlmanaged_deleter, Cvoid, (Ptr{Cvoid},))
end

function to_dlmanaged_tensor(view::TensorView)
    # Create DLManagedTensor
    dlmanaged = DLManagedTensor(
        view.dltensor,
        C_NULL,  # Will be set to pointer_from_objref
        _DLMANAGED_DELETER_CPTR[]
    )

    # Get pointer and store in pool to keep view alive
    dlmanaged_ptr = pointer_from_objref(dlmanaged)
    dlmanaged.manager_ctx = dlmanaged_ptr

    lock(_DLMANAGED_POOL_LOCK) do
        _DLMANAGED_POOL[dlmanaged_ptr] = (dlmanaged, view, view.source)
    end

    return Ptr{DLManagedTensor}(dlmanaged_ptr)
end

#=============================================================================
  Accessor Functions for TensorView
=============================================================================#

"""Get the element type of a TensorView."""
Base.eltype(::TensorView{T, S}) where {T, S} = T

"""Get the source type of a TensorView."""
source_type(::TensorView{T, S}) where {T, S} = S

"""Get the shape of a TensorView as a tuple."""
Base.size(view::TensorView) = Tuple(view.shape)

"""Get the number of dimensions."""
Base.ndims(view::TensorView) = length(view.shape)

"""Get the total number of elements."""
Base.length(view::TensorView) = prod(view.shape)

"""Get the device of a TensorView."""
device(view::TensorView) = view.dltensor.device

"""Get the dtype of a TensorView."""
dtype(view::TensorView) = view.dltensor.dtype

"""Check if the TensorView is contiguous (either C or F order)."""
function is_contiguous(view::TensorView)
    c_strides = _compute_c_contiguous_strides(view.shape)
    f_strides = _compute_f_contiguous_strides(view.shape)
    return view.strides == c_strides || view.strides == f_strides
end

"""Check if the TensorView is C-contiguous (row-major)."""
function is_c_contiguous(view::TensorView)
    return view.strides == _compute_c_contiguous_strides(view.shape)
end

"""Check if the TensorView is F-contiguous (column-major, Julia-style)."""
function is_f_contiguous(view::TensorView)
    return view.strides == _compute_f_contiguous_strides(view.shape)
end

"""Check if the TensorView is managed by Julia."""
is_julia_owned(view::TensorView) = view.ownership == JuliaOwned

"""Check if the TensorView is managed by TVM."""
is_tvm_owned(view::TensorView) = view.ownership == TVMOwned
