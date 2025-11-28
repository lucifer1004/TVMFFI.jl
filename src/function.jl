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
                        # GPU tensor - use _wrap_gpu_dltensor_view for proper GPU array type
                        julia_args[i] = _wrap_gpu_dltensor_view_from_ptr(tensor_ptr)
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

        # Identity optimization: if result is same Julia object as an input arg,
        # return the original argument directly instead of creating a new one.
        for i in 1:num_args
            if result === julia_args[i]
                if arg_raws[i].type_index == Int32(LibTVMFFI.kTVMFFIDLTensorPtr)
                    # DLTensorPtr: return borrowed view directly (no refcount to manage)
                    unsafe_store!(ret, arg_raws[i])
                    return 0
                elseif arg_raws[i].type_index == Int32(LibTVMFFI.kTVMFFITensor)
                    # TVMTensor: IncRef because caller expects owned reference
                    # (arg_raws[i] is borrowed, we need to return owned)
                    handle = reinterpret(LibTVMFFI.TVMFFIObjectHandle, arg_raws[i].data)
                    LibTVMFFI.TVMFFIObjectIncRef(handle)
                    unsafe_store!(ret, arg_raws[i])
                    return 0
                end
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
# Specialized method for zero arguments (avoids Vector allocations)
function (func::TVMFunction)()
    result = Ref{LibTVMFFI.TVMFFIAny}(LibTVMFFI.TVMFFIAny(
        Int32(LibTVMFFI.kTVMFFINone), 0, 0))

    ret = LibTVMFFI.TVMFFIFunctionCall(
        func.handle,
        Ptr{LibTVMFFI.TVMFFIAny}(C_NULL),
        Int32(0),
        Base.unsafe_convert(Ptr{LibTVMFFI.TVMFFIAny}, result)
    )
    check_call(ret)

    result_owned = TVMAny(result[])
    return take_value(result_owned)
end

# ============================================================================
# Small-N Specializations (1-10 arguments) - Stack allocation
# ============================================================================
# Like Rust's const STACK_LEN = 4, we specialize for small argument counts
# to avoid Vector{Any} allocations. Generated via macro.

"""
Internal helper: convert a single argument to TVMFFIAny and track array info.
Returns (raw_any, gc_ref, tensor_ptr_or_nothing)
"""
@inline function _convert_arg(arg)
    if arg isa AbstractArray && !(arg isa TensorView)
        # Use TensorView for ALL arrays (CPU and GPU) - much faster than TVMTensor
        # TensorView uses kTVMFFIDLTensorPtr (type_index=7), no C API call needed
        # GPU arrays are safe because GC.@preserve keeps them alive during FFI call
        view = TensorView(arg)
        any = TVMAny(view)
        return (raw_data(any), (any, view), (view.dltensor.data, arg))
    elseif arg isa TensorView
        any = TVMAny(arg)
        return (raw_data(any), (any, arg), (arg.dltensor.data, arg.source))
    else
        any = TVMAny(arg)
        return (raw_data(any), any, nothing)
    end
end

"""
    @_define_narg_method(N)

Internal macro to generate a specialized N-argument method for TVMFunction.
Uses quote blocks for cleaner metaprogramming.
"""
macro _define_narg_method(N)
    N = N::Int

    # Generate symbol names
    arg_syms = [Symbol("arg$i") for i in 1:N]
    raw_syms = [Symbol("raw$i") for i in 1:N]
    gc_syms = [Symbol("gc$i") for i in 1:N]
    tp_syms = [Symbol("tp$i") for i in 1:N]

    # Build conversion statements: (raw_i, gc_i, tp_i) = _convert_arg(arg_i)
    convert_stmts = [:(($(raw_syms[i]), $(gc_syms[i]), $(tp_syms[i])) = _convert_arg($(arg_syms[i])))
                     for i in 1:N]

    # Build raw tuple expression: (raw1, raw2, ...)
    raw_tuple_expr = Expr(:tuple, raw_syms...)

    # Build tp check: tp1 !== nothing || tp2 !== nothing || ...
    tp_check = if N == 1
        :($(tp_syms[1]) !== nothing)
    else
        foldl((a, tp) -> :($a || $tp !== nothing), tp_syms[2:end];
            init = :($(tp_syms[1]) !== nothing))
    end

    # Build DLTensorPtr identity checks (short-circuit return)
    dltensor_checks = [:($(tp_syms[i]) !== nothing && $(tp_syms[i])[1] == data_ptr &&
                         return $(tp_syms[i])[2])
                       for i in 1:N]

    # Build TVMTensor identity checks (with DecRef before return)
    tensor_checks = [quote
                         if $(tp_syms[i]) !== nothing && $(tp_syms[i])[1] == data_ptr
                             LibTVMFFI.TVMFFIObjectDecRef(handle)
                             return $(tp_syms[i])[2]
                         end
                     end
                     for i in 1:N]

    # Build the complete function using quote
    func_def = quote
        function (func::TVMFunction)($(arg_syms...))
            # Convert all arguments
            $(convert_stmts...)

            # Pack raw values into tuple (stack-allocated via Ref)
            args_raw = Ref($raw_tuple_expr)
            result = Ref{LibTVMFFI.TVMFFIAny}(
                LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFINone), 0, 0)
            )

            # Call TVM function
            local ret
            GC.@preserve $(arg_syms...) $(gc_syms...) args_raw begin
                ret = LibTVMFFI.TVMFFIFunctionCall(
                    func.handle,
                    Ptr{LibTVMFFI.TVMFFIAny}(pointer_from_objref(args_raw)),
                    Int32($N),
                    Base.unsafe_convert(Ptr{LibTVMFFI.TVMFFIAny}, result)
                )
            end
            check_call(ret)

            result_raw = result[]

            # Identity optimization: return original array if data pointer matches
            if $tp_check
                if result_raw.type_index == Int32(LibTVMFFI.kTVMFFIDLTensorPtr)
                    tensor_ptr = reinterpret(Ptr{DLTensor}, result_raw.data)
                    if tensor_ptr != C_NULL
                        dltensor = unsafe_load(tensor_ptr)
                        data_ptr = Ptr{Cvoid}(dltensor.data)
                        $(dltensor_checks...)
                    end
                elseif result_raw.type_index == Int32(LibTVMFFI.kTVMFFITensor)
                    handle = reinterpret(LibTVMFFI.TVMFFIObjectHandle, result_raw.data)
                    if handle != C_NULL
                        dltensor_ptr = get_dltensor_ptr(handle)
                        dltensor = unsafe_load(dltensor_ptr)
                        data_ptr = Ptr{Cvoid}(dltensor.data)
                        $(tensor_checks...)
                    end
                end
            end

            result_owned = TVMAny(result_raw)
            return take_value(result_owned)
        end
    end

    return esc(func_def)
end

# Generate specialized methods for 1-10 arguments
@_define_narg_method 1
@_define_narg_method 2
@_define_narg_method 3
@_define_narg_method 4
@_define_narg_method 5
@_define_narg_method 6
@_define_narg_method 7
@_define_narg_method 8
@_define_narg_method 9
@_define_narg_method 10

# ============================================================================
# General method for 11+ arguments (fallback to Vector allocation)
# ============================================================================
function (func::TVMFunction)(args...)
    num_args = length(args)

    # Convert arguments directly to raw TVMFFIAny for C API
    # gc_refs keeps TVMAny/TensorView/TVMTensor alive during call
    args_raw = Vector{LibTVMFFI.TVMFFIAny}(undef, num_args)
    gc_refs = Vector{Any}(undef, num_args)  # Keep TVMAny + views alive
    # For identity optimization: track (data_ptr, original_array) pairs
    # Use Vector instead of Dict - linear search is faster for small N and avoids allocation
    tensor_ptrs = Vector{Tuple{Ptr{Cvoid}, Any}}()

    for (i, arg) in enumerate(args)
        if arg isa AbstractArray && !(arg isa TensorView)
            # Use TensorView for ALL arrays (CPU and GPU) - much faster than TVMTensor
            # TensorView uses kTVMFFIDLTensorPtr (type_index=7), no C API call needed
            # GPU arrays are safe because GC.@preserve keeps them alive during FFI call
            view = TensorView(arg)
            any = TVMAny(view)
            args_raw[i] = raw_data(any)
            gc_refs[i] = (any, view)  # Keep both alive
            # Track data pointer for identity optimization
            push!(tensor_ptrs, (view.dltensor.data, arg))
        elseif arg isa TensorView
            any = TVMAny(arg)
            args_raw[i] = raw_data(any)
            gc_refs[i] = (any, arg)  # Keep both alive
            # Track TensorView data pointer too (source is the underlying array)
            push!(tensor_ptrs, (arg.dltensor.data, arg.source))
        else
            any = TVMAny(arg)
            args_raw[i] = raw_data(any)
            gc_refs[i] = any
        end
    end

    # Prepare result
    result = Ref{LibTVMFFI.TVMFFIAny}(LibTVMFFI.TVMFFIAny(
        Int32(LibTVMFFI.kTVMFFINone), 0, 0))

    # CRITICAL GC Safety:
    # We must preserve args, gc_refs, AND args_raw during the C call
    # args_raw holds the raw TVMFFIAny data passed to C
    local ret
    GC.@preserve args gc_refs args_raw begin
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

    # Identity optimization: if result points to same data as an input array,
    # return the original array directly (zero-copy)
    result_raw = result[]

    if !isempty(tensor_ptrs)
        if result_raw.type_index == Int32(LibTVMFFI.kTVMFFIDLTensorPtr)
            # CPU array: check DLTensor data pointer
            tensor_ptr = reinterpret(Ptr{DLTensor}, result_raw.data)
            if tensor_ptr != C_NULL
                dltensor = unsafe_load(tensor_ptr)
                data_ptr = Ptr{Cvoid}(dltensor.data)
                # Linear search - fast for small N, no allocation
                for (ptr, original_arr) in tensor_ptrs
                    if ptr == data_ptr
                        return original_arr
                    end
                end
            end
        elseif result_raw.type_index == Int32(LibTVMFFI.kTVMFFITensor)
            # GPU array: check TVMTensor's DLTensor data pointer
            # Directly compute DLTensor pointer from handle to avoid TVMTensor allocation
            handle = reinterpret(LibTVMFFI.TVMFFIObjectHandle, result_raw.data)
            if handle != C_NULL
                # DLTensor follows immediately after TVMFFIObject header
                dltensor_ptr = get_dltensor_ptr(handle)
                dltensor = unsafe_load(dltensor_ptr)
                data_ptr = Ptr{Cvoid}(dltensor.data)
                # Linear search - fast for small N, no allocation
                for (ptr, original_arr) in tensor_ptrs
                    if ptr == data_ptr
                        # Result is same tensor as input - return original array
                        # DecRef the returned handle since we won't use it
                        LibTVMFFI.TVMFFIObjectDecRef(handle)
                        return original_arr
                    end
                end
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

    # Only support CPU tensors here; GPU tensors are handled upstream.
    device_type = dltensor.device.device_type
    device_type != Int32(LibTVMFFI.kDLCPU) && return nothing

    T = dtype_to_julia_type(dltensor.dtype)
    ndim = Int(dltensor.ndim)
    shape_tuple = ndim > 0 ? ntuple(i -> Int(unsafe_load(dltensor.shape, i)), ndim) : ()
    strides_tuple = if dltensor.strides == C_NULL || ndim == 0
        nothing
    else
        ntuple(i -> Int(unsafe_load(dltensor.strides, i)), ndim)
    end

    data_ptr = Ptr{UInt8}(dltensor.data) + dltensor.byte_offset
    typed_ptr = Ptr{T}(data_ptr)

    # 0-D tensors -> reshape a scalar buffer entry
    if ndim == 0
        scalar = unsafe_wrap(Array, typed_ptr, 1)
        return reshape(scalar, ())
    end

    # 1-D tensors only skip copy if contiguous.
    if ndim == 1
        stride1 = strides_tuple === nothing ? 1 : strides_tuple[1]
        if stride1 == 1
            return unsafe_wrap(Array, typed_ptr, shape_tuple[1])
        end
    end

    if strides_tuple !== nothing
        f_strides = _compute_f_contiguous_strides(shape_tuple)
        if strides_tuple == f_strides
            return unsafe_wrap(Array, typed_ptr, Tuple(shape_tuple))
        end
    end

    # Layout mismatch (e.g., row-major) or arbitrary strided tensor → copy.
    shape_vec = collect(Int64.(shape_tuple))
    strides_vec = if strides_tuple === nothing
        _compute_c_contiguous_strides(shape_vec)
    else
        collect(Int64.(strides_tuple))
    end

    return _copy_strided_data(T, Ptr{Cvoid}(data_ptr), shape_vec, strides_vec)
end

"""
    _wrap_gpu_dltensor_view_from_ptr(tensor_ptr::Ptr{DLTensor})

Wrap a GPU DLTensor pointer as a native GPU array (CuArray, MtlArray, etc.).

This is used in callback contexts where we need a proper GPU array for
Julia operations. The caller guarantees data validity during the callback.
"""
function _wrap_gpu_dltensor_view_from_ptr(tensor_ptr::Ptr{DLTensor})
    dltensor = unsafe_load(tensor_ptr)

    # Get element type
    T = dtype_to_julia_type(dltensor.dtype)

    # Get device type for dispatch
    device_type = dltensor.device.device_type

    # Get shape and strides as tuples
    ndim = Int(dltensor.ndim)
    shape = if ndim > 0
        ntuple(i -> Int64(unsafe_load(dltensor.shape, i)), ndim)
    else
        ()
    end

    strides = if dltensor.strides != C_NULL && ndim > 0
        ntuple(i -> Int64(unsafe_load(dltensor.strides, i)), ndim)
    else
        # Default: C-contiguous → convert to Julia strides (column-major)
        _compute_c_contiguous_strides(shape)
    end

    # Dispatch to GPU backend extension
    return _wrap_gpu_dltensor_view(Val(device_type), T, dltensor.data, shape, strides)
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
