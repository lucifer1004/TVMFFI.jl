# TVMFFI.jl

Julia bindings for the TVM (Tensor Virtual Machine) FFI (Foreign Function Interface).

## Features

TVMFFI.jl provides a complete, idiomatic Julia interface to TVM's C API:

### ✅ Core Functionality
- **Device Management**: CPU, CUDA, OpenCL, Vulkan, Metal, ROCm support
- **Data Types**: Full DLPack integration with automatic type conversion
- **Zero-Copy Tensors**: Efficient data exchange via `DLTensorHolder`
- **Error Handling**: TVM errors automatically mapped to Julia exceptions
- **Strings & Bytes**: Small string optimization, heap allocation for large strings

### ✅ Function System
- **Call TVM Functions**: `get_global_func()` to retrieve and call TVM functions
- **Register Julia Functions**: `register_global_func()` exposes Julia functions to TVM
- **Automatic Conversion**: Seamless conversion between Julia and TVM types
- **Exception Safety**: Julia errors are properly translated to TVM errors

### ✅ Object System
- **Type Registration**: `register_object()` and `get_type_index()` for custom types
- **Reference Counting**: Automatic memory management via finalizers
- **Type Hierarchy**: Support for parent-child type relationships

### ✅ Module System
- **Load Modules**: `load_module()` to load compiled TVM modules
- **Query Functions**: `mod["function_name"]` or `get_function(mod, name)`
- **Module Caching**: Efficient global function caching

### 🔧 In Progress
- **Full Object Reflection**: Complete `@register_object` macro with field/method registration
- **System Library**: `system_lib()` for statically linked modules
- **Cross-language Tests**: Comparison tests against Python/Rust implementations

## Installation

```julia
# From the workspace root
cd TVMFFI
julia --project=. -e 'using Pkg; Pkg.instantiate()'
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

# Strings and error handling
s = TVMString("hello TVM")
@assert String(s) == "hello TVM"
```

### Calling TVM Functions

```julia
# Get a global function
func = get_global_func("my_tvm_function")
if func !== nothing
    result = func(arg1, arg2)
end
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

### Working with Arrays

```julia
# Zero-copy array conversion
x = Float32[1, 2, 3, 4, 5]
holder = from_julia_array(x)

# Use in function calls (automatic conversion)
result = some_tvm_func(x)  # Arrays auto-convert to DLTensorHolder
```

### Loading Modules

```julia
# Load a compiled module
mod = load_module("path/to/module.so")

# Get and call functions
my_func = mod["function_name"]
output = my_func(input1, input2)
```

### Custom Type Registration

```julia
# Define a custom type
struct MyCustomType
    value::Int
end

# Register it
idx = register_object("my_package.MyCustomType", MyCustomType)

# Look it up
idx2 = get_type_index("my_package.MyCustomType")
@assert idx == idx2
```

## Testing

Run the comprehensive test suite:

```julia
using Pkg
Pkg.test("TVMFFI")
```

Current test coverage: **84 tests** across:
- Version API (6 tests)
- Device creation (5 tests)
- Data type handling (8 tests)
- Data type handling  
- String/Bytes operations
- Error handling
- Type conversions
- Function registration and calls
- Object registration
- Tensor operations (CPU & GPU)
- Module API

## Examples

See the `examples/` directory for practical demonstrations:

### Quick Start
- **`basic_usage.jl`** - Comprehensive intro covering devices, dtypes, strings, errors

### Advanced Features
- **`list_types.jl`** - Explore registered TVM types (POD and Object types)
- **`test_gc_safety.jl`** - GC safety stress test with aggressive allocation

### Real-World Usage (requires compiled TVM modules)
- **`load_add_one.jl`** - Load and call CPU module (full walkthrough)
- **`load_add_one_cuda.jl`** - Load and call GPU module

**Note**: Most functionality is comprehensively tested in `test/runtests.jl` (84 tests). Examples focus on end-to-end scenarios and educational walkthroughs.

## Architecture

```
TVMFFI/
├── src/
│   ├── LibTVMFFI.jl          # Low-level C bindings
│   ├── TVMFFI.jl              # Main module
│   ├── device.jl              # Device abstractions
│   ├── dtype.jl               # Data type handling
│   ├── string.jl              # String/Bytes types
│   ├── error.jl               # Error handling
│   ├── object.jl              # Object system
│   ├── tensor.jl              # DLTensor support
│   ├── function.jl            # Function calls & registration
│   ├── gpuarrays_support.jl   # GPU array integration
│   └── module.jl              # Module loading
├── test/
│   └── runtests.jl            # Comprehensive test suite
└── examples/                  # Usage examples
```

## Design Philosophy

Following Linus Torvalds' principles:

1. **Good Taste**: Eliminate special cases through proper data structure design
2. **Simplicity**: Direct C API mapping with zero intermediate layers
3. **Practical**: Solve real problems, not theoretical ones
4. **Memory Safety**: Julia's GC + finalizers handle cleanup automatically

## Status

See [../STATUS.md](../STATUS.md) for detailed feature parity matrix with Python/Rust bindings.

**Current State**: ✅ **Bidirectional** - Can both call TVM and extend TVM with Julia code.

## Contributing

See [AGENTS.md](AGENTS.md) for development workflow and guidelines.

## License

Licensed under the Apache License 2.0. See the source file headers for details.

