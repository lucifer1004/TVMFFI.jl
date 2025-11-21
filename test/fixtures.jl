# Paths
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
    ensure_fixture_built(name::String)

Check if fixture exists, build if not.
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
        run(`cmake $(FIXTURES_DIR) -B $(FIXTURES_BUILD_DIR) -DCMAKE_BUILD_TYPE=RelWithDebInfo`)
        run(`cmake --build $(FIXTURES_BUILD_DIR) --config RelWithDebInfo`)

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

        @info "✓ Built: $name"
        return path

    catch e
        error("Failed to build fixture '$name': $e\nMake sure CMake and C++ compiler are installed.")
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
