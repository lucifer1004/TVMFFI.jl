# TVMFFI.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://lucifer1004.github.io/TVMFFI.jl/dev/)
[![Build Status](https://github.com/lucifer1004/TVMFFI.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lucifer1004/TVMFFI.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/lucifer1004/TVMFFI.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/lucifer1004/TVMFFI.jl)

Julia bindings for the TVM (TensorView Virtual Machine) FFI (Foreign Function Interface).

## Features

TVMFFI.jl provides a complete, idiomatic Julia interface to TVM's C API:

### ✅ Core Functionality
- **Device Management**: CPU, CUDA, OpenCL, Vulkan, Metal, ROCm support
- **Data Types**: Full DLPack integration with automatic type conversion
- **Zero-Copy TensorViews**: Efficient data exchange via `TensorView`
- **Error Handling**: TVM errors automatically mapped to Julia exceptions
- **Strings & Bytes**: Small string optimization, heap allocation for large strings

### ✅ Function System
- **Call TVM Functions**: `get_global_func()` to retrieve and call TVM functions
- **Register Julia Functions**: `register_global_func()` exposes Julia functions to TVM
- **Automatic Conversion**: Seamless conversion between Julia and TVM types
- **Exception Safety**: Julia errors are properly translated to TVM errors

### ✅ Object System
- **Type Registration**: `@register_object` macro for wrapping TVM types in Julia
- **Reflection API**: `get_type_info()`, `get_fields()`, `get_methods()` for introspection
- **Property Access**: Automatic field/method access via `obj.field` and `obj.method()`
- **Constructors**: Direct `TypeName(args...)` syntax for types with `__ffi_init__`
- **Reference Counting**: Automatic memory management via finalizers

### ✅ Module System
- **Load Modules**: `load_module()` to load compiled TVM modules
- **Query Functions**: `mod["function_name"]` or `get_function(mod, name)`
- **System Library**: `system_lib()` for statically linked modules
- **Module Introspection**: `inspect_source()`, `get_module_kind()`, `implements_function()`
- **Module Export**: `write_to_file()` to save compiled modules
- **Module Caching**: Efficient global function caching

## Installation

```julia
using Pkg
Pkg.add("TVMFFI")
```

Or for the latest development version:

```julia
Pkg.add(url="https://github.com/lucifer1004/TVMFFI.jl")
```

## Quick Start

### Basic Usage

```julia
using TVMFFI

# Check TVM FFI version
v = tvm_ffi_version()
println("TVM FFI Version: $v")  # e.g., "0.1.2"

# Create devices
cpu_dev = cpu(0)
cuda_dev = cuda(0)

# Data types
dt = DLDataType(Float32)
println(string(dt))  # "float32"
```

### Working with Arrays (Zero-Copy)

```julia
# 1. CPU Arrays
x = Float32[1, 2, 3, 4, 5]
holder = TensorView(x)

# 2. GPU Arrays (requires CUDA.jl, Metal.jl, etc.)
# Automatically detects device type and handles pointers correctly
using CUDA
x_gpu = CuArray(Float32[1, 2, 3])
holder_gpu = TensorView(x_gpu)

# 3. Call TVM functions
# Arrays are automatically converted to DLTensor
result = some_tvm_func(x_gpu)
```

### Loading Modules

```julia
# Load a compiled module
mod = load_module("path/to/module.so")

# Get and call functions
my_func = mod["function_name"]
output = my_func(input1, input2)
```

### Registering Julia Functions

```julia
# Define a Julia function
function my_add(x::Int64, y::Int64)
    return x + y
end

# Register it to TVM
register_global_func("julia.my_add", my_add)

# Call it from TVM
func = get_global_func("julia.my_add")
result = func(Int64(10), Int64(20))  # Returns 30
```

### Working with TVM Objects

```julia
# Register a TVM type for use in Julia
@register_object "testing.TestCxxClassBase" struct TestCxxClassBase end

# Create instances (if the type has __ffi_init__)
obj = TestCxxClassBase(Int64(42), Int32(10))

# Access fields via reflection
println(obj.v_i64)  # 42
println(obj.v_i32)  # 10

# Modify fields
obj.v_i64 = Int64(100)
obj.v_i32 = Int32(20)

# Type introspection
println(type_key(TestCxxClassBase))   # "testing.TestCxxClassBase"
println(type_index(TestCxxClassBase)) # Runtime type index
```

## Documentation

For full API documentation, see [Documentation](https://lucifer1004.github.io/TVMFFI.jl/dev/).

- **[API Reference](https://lucifer1004.github.io/TVMFFI.jl/dev/api/)**: Complete list of exported functions and types.

## Performance

FFI overhead has been carefully measured using [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl).

### FFI Overhead Summary

| Operation | Time | vs Julia | Allocations |
|-----------|------|----------|-------------|
| Pure Julia call | 1.1 ns | 1x | 0 |
| TVMAny(Int64) | 6 ns | 6x | 1 (32 B) |
| TensorView creation | 27 ns | 25x | 5 (208 B) |
| IncRef + DecRef | 6 ns | 5x | 0 |
| Dict lookup (callback) | 13 ns | 12x | 0 |
| **TVM func() empty** | **209 ns** | **190x** | 5 (160 B) |
| **TVM func(Int64)** | **272 ns** | **250x** | 10 (336 B) |
| TVM func(Array[64]) | 3.1 μs | 2800x | 62 (2.7 KB) |
| TVM func(TensorView[64]) | 3.0 μs | 2750x | 57 (2.5 KB) |
| TVM func(Array[4096]) | 19 μs | 17000x | 66 (35 KB) |

### Overhead Breakdown

For a typical scalar function call (~260 ns):

```
ccall + callback dispatch:  232 ns  (89.5%)  ← C-side overhead
Args Vector allocation:      12 ns  ( 4.7%)
TVMAny conversion:            7 ns  ( 2.6%)
Result Ref allocation:        6 ns  ( 2.5%)
raw_data extraction:          2 ns  ( 0.7%)
```

**Key insight**: 89.5% of overhead is in C-side callback dispatch, not Julia.

### Practical Impact

| Workload | FFI Overhead | Recommendation |
|----------|--------------|----------------|
| Inference (1ms+) | < 0.03% | ✅ Negligible |
| Micro-ops (1μs) | ~20-30% | ⚠️ Consider batching |
| Hot loops (100ns) | Dominates | ❌ Use raw ccall |

### Running Benchmarks

```bash
cd benchmarks
julia --project=. -e 'using Pkg; Pkg.develop(path=".."); Pkg.instantiate()'
julia --project=. ffi_overhead.jl
julia --project=. microbenchmarks.jl
```

## Architecture

```
TVMFFI/
├── src/
│   ├── LibTVMFFI.jl          # Low-level C bindings
│   ├── TVMFFI.jl             # Main module
│   ├── device.jl             # Device abstractions
│   ├── dtype.jl              # Data type handling
│   ├── string.jl             # String/Bytes types
│   ├── error.jl              # Error handling
│   ├── object.jl             # Object system
│   ├── tensor.jl             # DLTensor support
│   ├── function.jl           # Function calls & registration
│   ├── gpuarrays_support.jl  # GPU array integration (via DLPack.jl)
│   ├── dlpack.jl             # DLPack zero-copy tensor exchange
│   └── module.jl             # Module loading
├── ext/                      # Package extensions
│   ├── CUDAExt.jl            # Placeholder (DLPack.jl provides CUDA support)
│   ├── AMDGPUExt.jl          # AMD ROCm support
│   └── MetalExt.jl           # Apple Metal support
├── test/
│   └── runtests.jl           # Comprehensive test suite
└── examples/                 # Usage examples
```

### GPU Support

GPU device detection uses `DLPack.dldevice()` - no code duplication:
- **CUDA**: Handled by DLPack.jl's CUDAExt
- **AMD ROCm**: Handled by TVMFFI's AMDGPUExt  
- **Apple Metal**: Handled by TVMFFI's MetalExt

## Design Philosophy

Following Linus Torvalds' principles:

1. **Good Taste**: Eliminate special cases through proper data structure design
2. **Simplicity**: Direct C API mapping with zero intermediate layers
3. **Practical**: Solve real problems, not theoretical ones
4. **Memory Safety**: Julia's GC + finalizers handle cleanup automatically


## License

Licensed under the Apache License 2.0. See the source file headers for details.
