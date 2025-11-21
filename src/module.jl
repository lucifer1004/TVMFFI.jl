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

# Cache global functions at module initialization
# This avoids repeated lookups
const _module_loader = Ref{Union{TVMFunction, Nothing}}(nothing)
const _function_getter = Ref{Union{TVMFunction, Nothing}}(nothing)

"""
Initialize cached global functions.
Called from __init__() in main module.
"""
function _init_module_api()
    _module_loader[] = get_global_func("ffi.ModuleLoadFromFile")
    _function_getter[] = get_global_func("ffi.ModuleGetFunction")

    if _module_loader[] === nothing
        @warn "ffi.ModuleLoadFromFile not found - module loading unavailable"
    end
    if _function_getter[] === nothing
        @warn "ffi.ModuleGetFunction not found - function querying unavailable"
    end
end

"""
    TVMModule

Wrapper for TVM module objects.
"""
struct TVMModule
    handle::TVMObject

    TVMModule(obj::TVMObject) = new(obj)
end

"""
    load_module(path::AbstractString) -> TVMModule

Load a compiled TVM module from file.

# Example
```julia
mod = load_module("build/add_one_cpu.so")
```
"""
function load_module(path::AbstractString)
    loader = _module_loader[]
    if loader === nothing
        error("Module API not initialized - did __init__() run?")
    end

    mod_obj = loader(path)
    return TVMModule(mod_obj)
end

"""
    get_function(mod::TVMModule, name, query_imports=true) -> Union{TVMFunction, Nothing}

Get a function from the module by name.

Returns `nothing` if function not found.

# Example
```julia
mod = load_module("build/add_one_cpu.so")
add_one = get_function(mod, "add_one_cpu")
```
"""
function get_function(mod::TVMModule, name::AbstractString, query_imports::Bool = true)
    getter = _function_getter[]
    if getter === nothing
        error("Module API not initialized - did __init__() run?")
    end

    return getter(mod.handle, name, query_imports)
end

"""
    Base.getindex(mod::TVMModule, name::AbstractString) -> TVMFunction

Get a function from module using bracket notation (Python-style).

# Examples
```julia
mod = load_module("build/add_one_cpu.so")
add_one = mod["add_one_cpu"]  # Cleaner!
add_one(x, y)
```
"""
function Base.getindex(mod::TVMModule, name::AbstractString)
    func = get_function(mod, name, true)
    if func === nothing
        error("Function '$name' not found in module")
    end
    return func
end

function Base.show(io::IO, mod::TVMModule)
    print(io, "TVMModule(@", repr(UInt(mod.handle.handle)), ")")
end

"""
    system_lib(symbol_prefix::AbstractString = "") -> TVMModule

Get the system library module containing statically linked symbols.

The system library contains symbols that are registered via the C API
during static initialization. This is useful for:
- Accessing statically compiled TVM modules
- Testing with symbols registered at compile time
- Embedded systems without dynamic loading

# Arguments
- `symbol_prefix`: Optional prefix for symbol filtering (default: "")

# Examples
```julia
# Get the system library
mod = system_lib()

# With prefix filtering
test_mod = system_lib("testing.")  # Only symbols prefixed with "__tvm_ffi_testing."
```

# See Also
Python equivalent: `tvm_ffi.system_lib()`
Rust equivalent: `SystemLib::new()`
"""
function system_lib(symbol_prefix::AbstractString = "")
    system_lib_func = get_global_func("ffi.SystemLib")
    
    if system_lib_func === nothing
        error("ffi.SystemLib not found - system library support unavailable")
    end
    
    mod_obj = system_lib_func(String(symbol_prefix))
    return TVMModule(mod_obj)
end

"""
    write_to_file(mod::TVMModule, file_name::AbstractString, format::AbstractString = "")

Save a TVM module to a file.

# Arguments
- `mod`: The TVM module to save
- `file_name`: Output file path
- `format`: Optional format specifier (e.g., "so", "dll", "dylib")

# Examples
```julia
mod = load_module("my_module.so")
write_to_file(mod, "output.so", "so")
```

# See Also
Python equivalent: `mod.save(file_name, fmt)`
"""
function write_to_file(mod::TVMModule, file_name::AbstractString, format::AbstractString = "")
    write_func = get_global_func("ffi.ModuleWriteToFile")
    
    if write_func === nothing
        error("ffi.ModuleWriteToFile not found - module writing unavailable")
    end
    
    write_func(mod.handle, String(file_name), String(format))
    return nothing
end

"""
    inspect_source(mod::TVMModule, format::AbstractString = "") -> String

Inspect the source code of a module for debugging.

# Arguments
- `mod`: The TVM module to inspect
- `format`: Optional format specifier (e.g., "ll" for LLVM IR, "asm" for assembly)

# Examples
```julia
mod = load_module("my_module.so")
source = inspect_source(mod, "ll")
println(source)
```

# See Also
Python equivalent: `mod.get_source(fmt)`
"""
function inspect_source(mod::TVMModule, format::AbstractString = "")
    inspect_func = get_global_func("ffi.ModuleInspectSource")
    
    if inspect_func === nothing
        error("ffi.ModuleInspectSource not found - source inspection unavailable")
    end
    
    result = inspect_func(mod.handle, String(format))
    return String(result)
end

"""
    get_module_kind(mod::TVMModule) -> String

Get the kind/type of a module (e.g., "llvm", "cuda", "c").

# Examples
```julia
mod = load_module("my_module.so")
kind = get_module_kind(mod)
println("Module kind: ", kind)
```
"""
function get_module_kind(mod::TVMModule)
    kind_func = get_global_func("ffi.ModuleGetKind")
    
    if kind_func === nothing
        error("ffi.ModuleGetKind not found")
    end
    
    result = kind_func(mod.handle)
    return String(result)
end

"""
    implements_function(mod::TVMModule, name::AbstractString, query_imports::Bool = true) -> Bool

Check if a module implements a specific function.

# Examples
```julia
mod = load_module("my_module.so")
if implements_function(mod, "my_func")
    println("Function exists!")
end
```
"""
function implements_function(mod::TVMModule, name::AbstractString, query_imports::Bool = true)
    impl_func = get_global_func("ffi.ModuleImplementsFunction")
    
    if impl_func === nothing
        error("ffi.ModuleImplementsFunction not found")
    end
    
    result = impl_func(mod.handle, String(name), query_imports)
    return Bool(result)
end
