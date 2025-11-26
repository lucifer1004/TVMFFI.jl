#=
TVMFFI.jl Test Suite

This is the main test runner. Individual test files are organized by functionality:
- test_dtype_device.jl:   Device and DataType APIs
- test_string_bytes.jl:   TVM String and Bytes
- test_error.jl:          Error handling
- test_conversions.jl:    Type conversions (TVMAny/take_value/copy_value)
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

# Import internal APIs used in tests (not exported by default)
import TVMFFI: dtype_to_julia_type, check_call
import TVMFFI: TVMString, TVMBytes, TVMObject
import TVMFFI: DLTensor, DLDeviceType, DLDataTypeCode
import TVMFFI: TVMAny, TVMAnyView, take_value, copy_value, raw_data
import TVMFFI: register_object, get_type_index
import TVMFFI: get_type_info, get_fields, get_methods, FieldInfo, MethodInfo
import TVMFFI: get_field_value, set_field_value!, call_method, get_method_function
import TVMFFI: has_ffi_init, ffi_init
import TVMFFI: @register_object_simple
import TVMFFI: write_to_file, inspect_source, get_module_kind, implements_function
import TVMFFI: print_gpu_info, gpu_array_info

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
    
    # DLPack zero-copy
    include("test_dlpack.jl")

    # Module API
    include("test_modules.jl")

    # Memory safety
    include("test_gc_safety.jl")

    # Compiled fixtures
    include("test_fixtures.jl")
end

println("\n✓ All tests passed!")
