# TVMFFI Test Fixtures

This directory contains test fixtures (compiled TVM FFI modules) used by the test suite.

## Fixtures

### `add_one_cpu.cc`
A simple CPU function that performs element-wise addition: `y = x + 1`

**Function signature:**
```julia
add_one_cpu(x::DLTensorHolder{Float32}, y::DLTensorHolder{Float32})
```

**Usage in tests:**
```julia
using TVMFFI

# Load the module
mod = load_module("path/to/add_one_cpu.so")
add_one = mod["add_one_cpu"]

# Call the function
x = Float32[1, 2, 3, 4]
y = similar(x)
x_holder = from_julia_array(x)
y_holder = from_julia_array(y)
add_one(x_holder, y_holder)

@test y == Float32[2, 3, 4, 5]
```

## Building Fixtures

Fixtures are automatically built by the test suite on first run using TVMFFI_jll artifacts.

**Automatic build (recommended):**
```julia
using Pkg
Pkg.test("TVMFFI")  # Builds fixtures automatically
```

**Manual build via Julia:**
```julia
include("test/fixtures.jl")
using .TestFixtures
build_fixtures(force=true, verbose=true)
```

**Manual build via CMake:**
```bash
# CMake will automatically call Julia to get TVMFFI_jll paths
cd TVMFFI/fixtures
cmake . -B ../build/fixtures -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build ../build/fixtures
```

**Note:** CMake internally runs:
```cmake
execute_process(
  COMMAND julia --project=.. -e "using TVMFFI_jll; print(TVMFFI_jll.artifact_dir)"
)
```
So you just need Julia in your PATH!

The compiled libraries will be placed in `TVMFFI/build/`.

## Adding New Fixtures

1. Create a new `.cc` file in this directory
2. Implement your function using TVM FFI C++ API
3. Export it with `TVM_FFI_DLL_EXPORT_TYPED_FUNC`
4. Add compilation target to `CMakeLists.txt`
5. Update this README with usage example

