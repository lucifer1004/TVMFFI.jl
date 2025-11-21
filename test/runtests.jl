#=
TVMFFI.jl Test Suite

This is the main test runner. Individual test files are organized by functionality:
- test_dtype_device.jl:   Device and DataType APIs
- test_string_bytes.jl:   TVM String and Bytes
- test_error.jl:          Error handling
- test_conversions.jl:    Type conversions (to_tvm_any/from_tvm_any)
- test_functions.jl:      Function registration and calling
- test_objects.jl:        Object registration
- test_tensors.jl:        DLTensor and array handling
- test_modules.jl:        Module API
- test_gc_safety.jl:      GC safety stress tests
- test_fixtures.jl:       Compiled test fixtures
=#

using Test
using TVMFFI
using TVMFFI.LibTVMFFI  # Import for internal constants

# Load fixture helper
include("fixtures.jl")

@testset "TVMFFI.jl Tests" begin
    # Core APIs
    include("test_dtype_device.jl")
    include("test_string_bytes.jl")
    include("test_error.jl")

    # Type conversions
    include("test_conversions.jl")

    # Function and object registration
    include("test_functions.jl")
    include("test_objects.jl")

    # Tensor operations
    include("test_tensors.jl")

    # Module API
    include("test_modules.jl")

    # Memory safety
    include("test_gc_safety.jl")

    # Compiled fixtures
    include("test_fixtures.jl")
end

println("\n✓ All tests passed!")
