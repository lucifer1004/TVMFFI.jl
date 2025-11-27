#=
Test fixture utilities - shared implementation

This file provides a test-specific wrapper around the shared fixtures utilities.
The actual build logic is in the shared implementation to avoid duplication.
=#

# Paths (relative to test/ directory)
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
    ensure_fixture_built(name::String) -> String

Check if fixture exists, build if not. Returns path to fixture.

For tests, we suppress verbose output to keep test logs clean.
"""
function ensure_fixture_built(name::String)
    path = fixture_path(name)

    # Already built?
    if isfile(path)
        return path
    end

    @info "Building fixture: $name"

    # Create build directory
    mkpath(FIXTURES_BUILD_DIR)

    # Run CMake (it will call Julia to find TVMFFI_jll)
    try
        # Suppress output for clean test logs
        run(pipeline(
            `cmake $(FIXTURES_DIR) -B $(FIXTURES_BUILD_DIR) -DCMAKE_BUILD_TYPE=RelWithDebInfo`,
            devnull))
        run(pipeline(
            `cmake --build $(FIXTURES_BUILD_DIR) --config RelWithDebInfo`, devnull))

        # Copy to build/ for easy access
        for lib_file in readdir(FIXTURES_BUILD_DIR)
            if endswith(lib_file, ".so") || endswith(lib_file, ".dylib") ||
               endswith(lib_file, ".dll")
                cp(joinpath(FIXTURES_BUILD_DIR, lib_file),
                    joinpath(BUILD_DIR, lib_file); force = true)
            end
        end

        if !isfile(path)
            error("Build succeeded but fixture not found: $path")
        end

        @info "✓ Built: $name"
        return path

    catch e
        error("""
        Failed to build fixture '$name': $e

        Requirements:
          • CMake (version ≥ 3.20)
          • C++ compiler (g++/clang++)
          • CUDA toolkit (for CUDA fixtures)
        """)
    end
end

"""
    load_fixture(name::String) -> TVMModule

Load a test fixture, building if necessary.
"""
function load_fixture(name::String)
    path = ensure_fixture_built(name)
    return TVMFFI.load_module(path)
end
