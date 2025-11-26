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

function callback_deleter(resource_handle::Ptr{Cvoid})
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
        # We use TVMAnyView to make ownership explicit, then copy_value to own them
        julia_args = Vector{Any}(undef, num_args)
        for i in 1:num_args
            # Pointer arithmetic to get i-th argument
            arg_ptr = args + (i - 1) * sizeof(LibTVMFFI.TVMFFIAny)
            arg_raw = unsafe_load(arg_ptr)
            # Create view (borrowed) then copy to own the value
            view = TVMAnyView(arg_raw)
            julia_args[i] = copy_value(view)
        end

        # Call function
        result = func(julia_args...)

        # Convert result to TVMAny
        # NOTE: For arrays, TVMAny(array) returns (any, holder) tuple
        # The holder must be kept alive during the store
        result_holder = nothing
        result_any = if result isa AbstractArray && !(result isa DLTensorHolder)
            # Create holder and TVMAny
            result_holder = DLTensorHolder(result)
            TVMAny(result_holder)
        else
            TVMAny(result)
        end

        # Write result to ret pointer
        # CRITICAL: Must preserve result_holder during this store
        GC.@preserve result_holder result_any begin
            unsafe_store!(ret, raw_data(result_any))
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
end

"""
    TVMFunction

Wrapper for TVM function objects with automatic argument conversion.
"""
mutable struct TVMFunction
    handle::LibTVMFFI.TVMFFIObjectHandle

    """
        TVMFunction(handle; borrowed=true)

    Create a TVMFunction from a raw handle.

    # Arguments
    - `handle`: The raw function handle
    - `borrowed`: Reference semantics
      - `borrowed=true` (default): Borrowed reference, increment refcount (safe)
      - `borrowed=false`: Owned reference, take without IncRef (C gave us ownership)
    """
    function TVMFunction(handle::LibTVMFFI.TVMFFIObjectHandle; borrowed::Bool = true)
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

Arrays are automatically converted to DLTensorHolder for convenience.
Pre-created holders can be passed for performance optimization.

# Examples
```julia
# Simple - auto-conversion
func(x, y)

# Optimized - reuse holders
holder = DLTensorHolder(x)
for i in 1:1000000
    func(holder)
end
```
"""
function (func::TVMFunction)(args...)
    num_args = length(args)
    
    # Convert arguments to TVMAny
    # Arrays need special handling - they return (any, holder) tuples
    args_any = Vector{TVMAny}(undef, num_args)
    holders = Vector{Any}(undef, num_args)  # Keep holders alive
    
    for (i, arg) in enumerate(args)
        if arg isa AbstractArray && !(arg isa DLTensorHolder)
            holder = DLTensorHolder(arg)
            args_any[i] = TVMAny(holder)
            holders[i] = holder  # Keep holder alive
        elseif arg isa DLTensorHolder
            args_any[i] = TVMAny(arg)
            holders[i] = arg  # Keep holder alive
        else
            args_any[i] = TVMAny(arg)
            holders[i] = nothing
        end
    end
    
    # Extract raw TVMFFIAny for C API
    args_raw = [raw_data(a) for a in args_any]

    # Prepare result
    result = Ref{LibTVMFFI.TVMFFIAny}(LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFINone), 0, 0))

    # CRITICAL GC Safety:
    # We must preserve args_any and holders during the C call
    # - args_any: managed TVMAny objects
    # - holders: DLTensorHolders with array data
    local ret
    GC.@preserve args args_any holders begin
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

    # Convert result back to Julia type
    # Function return: C gives us ownership, use take_value
    result_owned = TVMAny(result[])
    julia_result = take_value(result_owned)

    # No manual cleanup needed! 
    # TVMAny finalizers will automatically DecRef when GC collects args_any

    return julia_result
end

# ============================================================================
# Private Helper Functions for DLTensor Conversion
# ============================================================================

function Base.show(io::IO, func::TVMFunction)
    print(io, "TVMFunction(@", repr(UInt(func.handle)), ")")
end
