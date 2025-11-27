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

# Global registry to keep Julia functions alive when passed to C
const _callback_registry = Dict{Ptr{Cvoid}, Any}()
const _callback_lock = ReentrantLock()

# C function pointers for callbacks
const _safe_call_ptr = Ref{Ptr{Cvoid}}(C_NULL)
const _deleter_ptr = Ref{Ptr{Cvoid}}(C_NULL)

# _julia_is_exiting is defined in tensor.jl (shared flag)
const _callback_atexit_registered = Ref(false)

"""
Cleanup callback registry at exit.
This prevents TVM from calling back into Julia during shutdown.
"""
function _cleanup_callbacks_at_exit()
    _julia_is_exiting[] = true
    # Clear registry - don't need locks during exit
    empty!(_callback_registry)
end

function callback_deleter(resource_handle::Ptr{Cvoid})
    # Skip if Julia is exiting - runtime may be in inconsistent state
    _julia_is_exiting[] && return nothing

    lock(_callback_lock) do
        delete!(_callback_registry, resource_handle)
    end
    nothing
end

function safe_call(
        resource_handle::Ptr{Cvoid},
        args::Ptr{LibTVMFFI.TVMFFIAny},
        num_args::Cint,
        ret::Ptr{LibTVMFFI.TVMFFIAny}
)::Cint
    try
        # Look up function
        func_ref = lock(_callback_lock) do
            get(_callback_registry, resource_handle, nothing)
        end

        if func_ref === nothing
            error("Callback function not found in registry (handle: $resource_handle)")
        end

        func = func_ref[]

        # Convert arguments
        # NOTE: args from C are borrowed references (views)
        # For DLTensorPtr, we use unsafe_wrap to create a zero-copy view instead of copying.
        # This is safe because caller's GC.@preserve keeps data alive during callback.
        julia_args = Vector{Any}(undef, num_args)
        arg_raws = Vector{LibTVMFFI.TVMFFIAny}(undef, num_args)

        for i in 1:num_args
            arg_ptr = args + (i - 1) * sizeof(LibTVMFFI.TVMFFIAny)
            arg_raw = unsafe_load(arg_ptr)
            arg_raws[i] = arg_raw

            if arg_raw.type_index == Int32(LibTVMFFI.kTVMFFIDLTensorPtr)
                # Zero-copy: wrap DLTensor data as Julia array view
                tensor_ptr = reinterpret(Ptr{DLTensor}, arg_raw.data)
                if tensor_ptr != C_NULL
                    wrapped = _wrap_dltensor_view(tensor_ptr)
                    if wrapped !== nothing
                        julia_args[i] = wrapped
                    else
                        # GPU tensor - fall back to copy_value
                        view = TVMAnyView(arg_raw)
                        julia_args[i] = copy_value(view)
                    end
                else
                    julia_args[i] = nothing
                end
            else
                view = TVMAnyView(arg_raw)
                julia_args[i] = copy_value(view)
            end
        end

        # Call function
        result = func(julia_args...)

        # Identity optimization for CPU arrays only (DLTensorPtr).
        # 
        # NOTE: GPU identity optimization is NOT possible because from_dlpack()
        # creates a NEW Julia array object each time. result === julia_args[i]
        # will always be false even if the underlying data is the same.
        for i in 1:num_args
            if result === julia_args[i] &&
               arg_raws[i].type_index == Int32(LibTVMFFI.kTVMFFIDLTensorPtr)
                # DLTensorPtr: return borrowed view directly (no refcount to manage)
                unsafe_store!(ret, arg_raws[i])
                return 0
            end
        end

        # Convert result to TVMAny
        # NOTE: For arrays, use TVMTensor for zero-copy return with proper lifecycle.
        result_any = if result isa StridedArray && !(result isa TensorView)
            # Use TVMTensor for zero-copy + reference counting
            tensor = TVMTensor(result)
            TVMAny(tensor)
        elseif result isa TensorView
            # TensorView has no refcounting - use as-is
            TVMAny(result)
        else
            TVMAny(result)
        end

        # Transfer ownership to return value
        GC.@preserve result_any begin
            unsafe_store!(ret, transfer_ownership!(result_any))
        end

        return 0
    catch e
        # Handle exception
        msg = sprint(showerror, e, catch_backtrace())

        # Create TVM error
        # CRITICAL: Must preserve strings during C API call to prevent GC
        kind_str = "RuntimeError"
        GC.@preserve kind_str msg begin
            kind_bytes = LibTVMFFI.TVMFFIByteArray(
                Ptr{UInt8}(pointer(kind_str)), UInt(sizeof(kind_str))
            )
            msg_bytes = LibTVMFFI.TVMFFIByteArray(
                Ptr{UInt8}(pointer(msg)), UInt(sizeof(msg))
            )
            # No backtrace for now (empty byte array)
            bt_bytes = LibTVMFFI.TVMFFIByteArray(C_NULL, 0)

            err_ret,
            err_handle = LibTVMFFI.TVMFFIErrorCreate(kind_bytes, msg_bytes, bt_bytes)

            if err_ret == 0 && err_handle != C_NULL
                # SetRaised only sets TLS, doesn't take ownership
                # We created the error, so we own it and must DecRef
                LibTVMFFI.TVMFFIErrorSetRaised(err_handle)
                LibTVMFFI.TVMFFIObjectDecRef(err_handle)
            end
        end

        return -1
    end
end

function _init_function_api()
    _safe_call_ptr[] = @cfunction(safe_call, Cint,
        (Ptr{Cvoid}, Ptr{LibTVMFFI.TVMFFIAny}, Cint, Ptr{LibTVMFFI.TVMFFIAny}))
    _deleter_ptr[] = @cfunction(callback_deleter, Cvoid, (Ptr{Cvoid},))

    # Register atexit handler to prevent callbacks during shutdown
    if !_callback_atexit_registered[]
        atexit(_cleanup_callbacks_at_exit)
        _callback_atexit_registered[] = true
    end
end

"""
    TVMFunction

Wrapper for TVM function objects with automatic argument conversion.
"""
mutable struct TVMFunction
    handle::LibTVMFFI.TVMFFIObjectHandle

    """
        TVMFunction(handle; borrowed)

    Create a TVMFunction from a raw handle. Internal API - users should not call directly.

    # Arguments
    - `handle`: The raw function handle
    - `borrowed`: Reference semantics (REQUIRED - no default to prevent misuse)
      - `borrowed=true`: Borrowed reference, increment refcount
      - `borrowed=false`: Owned reference, take without IncRef (C gave us ownership)
    """
    function TVMFunction(handle::LibTVMFFI.TVMFFIObjectHandle; borrowed::Bool)
        if handle == C_NULL
            error("Cannot create TVMFunction from NULL handle")
        end

        # Copy reference if borrowed
        if borrowed
            LibTVMFFI.TVMFFIObjectIncRef(handle)
        end

        func = new(handle)

        # Finalizer
        finalizer(func) do f
            if f.handle != C_NULL
                LibTVMFFI.TVMFFIObjectDecRef(f.handle)
            end
        end

        return func
    end
end

# Type system integration for TVMFunction
type_index(func::TVMFunction) = LibTVMFFI.TVMFFIObjectGetTypeIndex(func.handle)
type_index(::Type{TVMFunction}) = Int32(LibTVMFFI.kTVMFFIFunction)
type_key(::Type{TVMFunction}) = "ffi.Function"

"""
    get_global_func(name::AbstractString) -> Union{TVMFunction, Nothing}

Get a global function by name.

# Arguments
- `name::AbstractString`: The name of the global function

# Returns
- `TVMFunction` if the function exists
- `nothing` if the function does not exist

# Examples
```julia
func = get_global_func("my_custom_function")
if func !== nothing
    result = func(arg1, arg2)
end
```
"""
function get_global_func(name::AbstractString)
    name_str = String(name)

    local ret, handle
    GC.@preserve name_str begin
        byte_array = LibTVMFFI.TVMFFIByteArray(
            Ptr{UInt8}(pointer(name_str)), UInt(sizeof(name_str))
        )
        ret, handle = LibTVMFFI.TVMFFIFunctionGetGlobal(byte_array)
    end

    if ret != 0
        # Check if error is "function not found" or something else
        error_handle = LibTVMFFI.TVMFFIErrorMoveFromRaised()
        if error_handle != C_NULL
            throw(TVMError(error_handle; borrowed = false))  # C API transferred ownership
        else
            # No error but non-zero return
            error("Failed to get global function '$name' with code $ret")
        end
    end

    if handle == C_NULL
        return nothing
    end

    # C API returns a new reference, so we take ownership without IncRef
    return TVMFunction(handle; borrowed = false)
end

"""
    register_global_func(name::AbstractString, func::Function; override::Bool=false)

Register a Julia function as a global TVM function.

# Arguments
- `name`: Global function name (e.g., "my_pkg.my_func")
- `func`: Julia function to register
- `override`: Whether to override existing function
"""
function register_global_func(name::AbstractString, func::Function; override::Bool = false)
    # Create a reference to the function to keep it alive
    func_ref = Ref{Any}(func)

    # Use the pointer to the Ref as the resource handle
    # This is unique and stable as long as the Ref is alive
    resource_handle = Base.unsafe_convert(Ptr{Cvoid}, func_ref)

    # Register in global registry
    lock(_callback_lock) do
        _callback_registry[resource_handle] = func_ref
    end

    # Create TVM function
    ret,
    func_handle = LibTVMFFI.TVMFFIFunctionCreate(
        resource_handle,
        _safe_call_ptr[],
        _deleter_ptr[]
    )

    check_call(ret)

    if func_handle == C_NULL
        error("Failed to create TVM function")
    end

    # Register global
    name_str = String(name)

    local ret
    GC.@preserve name_str begin
        name_bytes = LibTVMFFI.TVMFFIByteArray(
            Ptr{UInt8}(pointer(name_str)), UInt(sizeof(name_str))
        )
        ret = LibTVMFFI.TVMFFIFunctionSetGlobal(name_bytes, func_handle, Int32(override))
    end

    check_call(ret)

    # We can DecRef the function handle now, as TVM global registry holds a reference
    LibTVMFFI.TVMFFIObjectDecRef(func_handle)

    return nothing
end

"""
    (func::TVMFunction)(args...) -> Any

Call a TVM function with arguments.

Arrays are automatically converted to TensorView for convenience.
Pre-created views can be passed for performance optimization.

# Examples
```julia
# Simple - auto-conversion
func(x, y)

# Optimized - reuse views
view = TensorView(x)
for i in 1:1000000
    func(view)
end
```
"""
function (func::TVMFunction)(args...)
    num_args = length(args)

    # Convert arguments to TVMAny
    # Arrays need special handling - they return (any, view) tuples
    # We also track array data pointers for identity optimization
    args_any = Vector{TVMAny}(undef, num_args)
    views = Vector{Any}(undef, num_args)  # Keep views alive
    # Map: DLTensor data pointer → (index, original array)
    tensor_ptr_to_arg = Dict{Ptr{Cvoid}, Tuple{Int, Any}}()

    for (i, arg) in enumerate(args)
        if arg isa AbstractArray && !(arg isa TensorView)
            # Check if GPU array - use TVMTensor (with refcount) for GPU, TensorView for CPU
            # GPU arrays need kTVMFFITensor type for proper from_dlpack handling in callbacks
            device = dldevice(arg)
            if device.device_type != Int32(LibTVMFFI.kDLCPU)
                # GPU array - use TVMTensor for proper DLPack + refcounting
                tensor = TVMTensor(arg)
                args_any[i] = TVMAny(tensor)
                views[i] = tensor  # Keep tensor alive
                # Track data pointer for identity optimization
                tensor_ptr_to_arg[Ptr{Cvoid}(UInt(pointer(arg)))] = (i, arg)
            else
                # CPU array - use TensorView for zero-copy
                view = TensorView(arg)
                args_any[i] = TVMAny(view)
                views[i] = view  # Keep view alive
                # Track data pointer for identity optimization
                tensor_ptr_to_arg[view.dltensor.data] = (i, arg)
            end
        elseif arg isa TensorView
            args_any[i] = TVMAny(arg)
            views[i] = arg  # Keep view alive
            # Track TensorView data pointer too (source is the underlying array)
            tensor_ptr_to_arg[arg.dltensor.data] = (i, arg.source)
        else
            args_any[i] = TVMAny(arg)
            views[i] = nothing
        end
    end

    # Extract raw TVMFFIAny for C API
    args_raw = [raw_data(a) for a in args_any]

    # Prepare result
    result = Ref{LibTVMFFI.TVMFFIAny}(LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFINone), 0, 0))

    # CRITICAL GC Safety:
    # We must preserve args_any, views, AND args_raw during the C call
    # args_raw holds the raw TVMFFIAny data passed to C
    local ret
    GC.@preserve args args_any views args_raw begin
        ret = if num_args > 0
            LibTVMFFI.TVMFFIFunctionCall(
                func.handle,
                pointer(args_raw),
                Int32(num_args),
                Base.unsafe_convert(Ptr{LibTVMFFI.TVMFFIAny}, result)
            )
        else
            LibTVMFFI.TVMFFIFunctionCall(
                func.handle,
                Ptr{LibTVMFFI.TVMFFIAny}(C_NULL),
                Int32(0),
                Base.unsafe_convert(Ptr{LibTVMFFI.TVMFFIAny}, result)
            )
        end
    end

    check_call(ret)

    # Identity optimization: if result is kTVMFFIDLTensorPtr (CPU array) pointing to 
    # one of our input arrays, return the original array directly (zero-copy)
    #
    # NOTE: GPU arrays (kTVMFFITensor) do NOT support identity optimization due to
    # complex reference counting semantics. See AGENTS.md for details.
    result_raw = result[]

    if result_raw.type_index == Int32(LibTVMFFI.kTVMFFIDLTensorPtr) &&
       !isempty(tensor_ptr_to_arg)
        tensor_ptr = reinterpret(Ptr{DLTensor}, result_raw.data)
        if tensor_ptr != C_NULL
            dltensor = unsafe_load(tensor_ptr)
            data_ptr = Ptr{Cvoid}(dltensor.data)
            if haskey(tensor_ptr_to_arg, data_ptr)
                # Result points to same data as input - return original array
                _, original_arr = tensor_ptr_to_arg[data_ptr]
                return original_arr
            end
        end
    end

    # Convert result back to Julia type
    result_owned = TVMAny(result_raw)
    julia_result = take_value(result_owned)

    return julia_result
end

# ============================================================================
# Private Helper Functions for DLTensor Conversion
# ============================================================================

"""
    _wrap_dltensor_view(tensor_ptr::Ptr{DLTensor}) -> AbstractArray

Create a zero-copy Julia array view from a DLTensor pointer.
The returned array shares memory with the DLTensor and must not outlive it.

This is used for callback arguments where the caller guarantees data lifetime.

# GPU Support
For GPU tensors (CUDA, ROCm, etc.), returns `nothing` to signal that the caller
should fall back to the standard conversion path. GPU arrays cannot be wrapped
with `unsafe_wrap` like CPU arrays.
"""
function _wrap_dltensor_view(tensor_ptr::Ptr{DLTensor})
    dltensor = unsafe_load(tensor_ptr)

    # Check device type - only CPU supports zero-copy unsafe_wrap
    device_type = dltensor.device.device_type
    if device_type != Int32(LibTVMFFI.kDLCPU)
        # GPU device - cannot use unsafe_wrap, return nothing to signal fallback
        return nothing
    end

    # Get element type
    T = dtype_to_julia_type(dltensor.dtype)

    # Get shape
    ndim = Int(dltensor.ndim)
    shape = ntuple(i -> Int(unsafe_load(dltensor.shape, i)), ndim)

    # Create array view (zero-copy)
    data_ptr = Ptr{T}(dltensor.data)

    # Check if contiguous (strides == NULL means C-contiguous)
    if dltensor.strides == C_NULL
        # C-contiguous - can use simple unsafe_wrap
        if ndim == 1
            return unsafe_wrap(Array, data_ptr, shape[1])
        else
            # Multi-dim: need to handle row-major to column-major conversion
            total = prod(shape)
            flat = unsafe_wrap(Array, data_ptr, total)
            return reshape(flat, reverse(shape)...)
        end
    else
        # Has explicit strides - check if actually contiguous
        strides = ntuple(i -> Int(unsafe_load(dltensor.strides, i)), ndim)

        # Check for F-contiguous (column-major, stride[1]=1)
        is_f_contiguous = ndim == 0 || (strides[1] == 1 &&
                           all(i -> strides[i] == strides[i - 1] * shape[i - 1], 2:ndim))

        # Check for C-contiguous (row-major, stride[end]=1)
        is_c_contiguous = ndim == 0 || (strides[end] == 1 &&
                           all(i -> strides[i] == strides[i + 1] * shape[i + 1], 1:(ndim - 1)))

        if is_f_contiguous && ndim == 1
            # 1D F-contiguous with stride=1 - can use unsafe_wrap
            return unsafe_wrap(Array, data_ptr, shape[1])
        elseif is_f_contiguous
            # Multi-dim F-contiguous - can use unsafe_wrap with reshape
            total = prod(shape)
            flat = unsafe_wrap(Array, data_ptr, total)
            return reshape(flat, shape...)
        else
            # Non-contiguous - must copy with strides
            result = Array{T}(undef, shape...)
            _copy_strided!(result, data_ptr, collect(Int64.(shape)), collect(Int64.(strides)))
            return result
        end
    end
end

function Base.show(io::IO, func::TVMFunction)
    print(io, "TVMFunction(@", repr(UInt(func.handle)), ")")
end

#------------------------------------------------------------
# Section: Reflection Support (depends on MethodInfo from object.jl)
#------------------------------------------------------------

"""
    get_method_function(method::MethodInfo) -> TVMFunction

Get the TVMFunction for a MethodInfo. Creates a borrowed reference.
This function is defined here because it depends on TVMFunction.
"""
function get_method_function(method::MethodInfo)
    if method.method_handle == C_NULL
        error("Method '$(method.name)' has no function handle")
    end
    return TVMFunction(method.method_handle; borrowed = true)
end
