#=
Shared fixture utilities for examples and tests
=#

using TVMFFI

# Paths (relative to this file in examples/)
const FIXTURES_DIR = joinpath(@__DIR__, "..", "fixtures")
const BUILD_DIR = joinpath(@__DIR__, "..", "build")
const FIXTURES_BUILD_DIR = joinpath(BUILD_DIR, "fixtures")

"""
    fixture_path(name::String) -> String

Get path to a fixture library file.
"""
function fixture_path(name::String)
    ext = Sys.islinux() ? ".so" : Sys.isapple() ? ".dylib" : ".dll"
    return joinpath(BUILD_DIR, name * ext)
end

"""
    ensure_fixture_built(name::String; verbose::Bool=true) -> String

Check if fixture exists, build if not. Returns path to fixture.

# Arguments
- `name`: Name of fixture (e.g., "add_one_cpu", "add_one_cuda")
- `verbose`: Print build messages (default: true)

# Returns
Path to the built fixture library.

# Example
```julia
path = ensure_fixture_built("add_one_cpu")
module_loader = get_global_func("ffi.ModuleLoadFromFile")
tvm_module = module_loader(path)
```
"""
function ensure_fixture_built(name::String; verbose::Bool=true)
    path = fixture_path(name)

    # Already built?
    if isfile(path)
        verbose && @info "✓ Fixture already built: $name"
        return path
    end

    verbose && @info "Building fixture: $name (this may take a minute on first run...)"

    # Create build directory
    mkpath(FIXTURES_BUILD_DIR)

    # Run CMake (it will call Julia to find TVMFFI_jll)
    try
        # Suppress output unless verbose
        redirect = verbose ? identity : pipeline
        
        if verbose
            println("   → Running CMake configure...")
            run(`cmake $(FIXTURES_DIR) -B $(FIXTURES_BUILD_DIR) -DCMAKE_BUILD_TYPE=RelWithDebInfo`)
            println("   → Building...")
            run(`cmake --build $(FIXTURES_BUILD_DIR) --config RelWithDebInfo`)
        else
            run(pipeline(`cmake $(FIXTURES_DIR) -B $(FIXTURES_BUILD_DIR) -DCMAKE_BUILD_TYPE=RelWithDebInfo`, devnull))
            run(pipeline(`cmake --build $(FIXTURES_BUILD_DIR) --config RelWithDebInfo`, devnull))
        end

        # Copy to build/ for easy access
        for lib_file in readdir(FIXTURES_BUILD_DIR)
            if endswith(lib_file, ".so") || endswith(lib_file, ".dylib") ||
               endswith(lib_file, ".dll")
                cp(joinpath(FIXTURES_BUILD_DIR, lib_file), joinpath(BUILD_DIR, lib_file); force = true)
            end
        end

        if !isfile(path)
            error("Build succeeded but fixture not found: $path")
        end

        verbose && @info "✓ Successfully built: $name → $path"
        return path

    catch e
        error("""
        Failed to build fixture '$name': $e
        
        Requirements:
          • CMake (version ≥ 3.20)
          • C++ compiler (g++/clang++)
          • CUDA toolkit (for CUDA fixtures)
          
        Please install missing dependencies and try again.
        """)
    end
end

"""
    load_fixture(name::String; verbose::Bool=true) -> TVMModule

Load a test fixture, building if necessary.

# Arguments
- `name`: Name of fixture (e.g., "add_one_cpu", "add_one_cuda")
- `verbose`: Print build messages (default: true)

# Returns
Loaded TVM module.

# Example
```julia
tvm_module = load_fixture("add_one_cpu")
add_one = get_function(tvm_module, "add_one_cpu")
add_one(x, y)
```
"""
function load_fixture(name::String; verbose::Bool=true)
    path = ensure_fixture_built(name; verbose=verbose)
    
    # Load using TVM FFI
    module_loader = get_global_func("ffi.ModuleLoadFromFile")
    if module_loader === nothing
        error("ffi.ModuleLoadFromFile not found! TVM FFI runtime may not be loaded correctly.")
    end
    
    return module_loader(path)
end

