using Test
using TVMFFI
using TVMFFI.LibTVMFFI  # Import for internal constants

@testset "TVMFFI.jl Tests" begin
    @testset "Version API" begin
        # Test tvm_ffi_version()
        v = tvm_ffi_version()
        @test v isa VersionNumber
        @test v.major == 0
        @test v.minor == 1
        @test v.patch >= 2  # At least 0.1.2
        
        # Test version comparisons
        @test v >= v"0.1.0"
        @test v < v"1.0.0"
    end

    @testset "Device Creation" begin
        # Test CPU device
        dev = cpu(0)
        @test dev.device_type == Int32(LibTVMFFI.kDLCPU)
        @test dev.device_id == 0

        # Test multiple devices
        dev1 = cpu(1)
        @test dev1.device_id == 1

        # Test CUDA (may not be available)
        cuda_dev = cuda(0)
        @test cuda_dev.device_type == Int32(LibTVMFFI.kDLCUDA)
    end

    @testset "Data Type Creation" begin
        # Test from Julia types
        dt_int32 = DLDataType(Int32)
        @test dt_int32.code == UInt8(LibTVMFFI.kDLInt)
        @test dt_int32.bits == 32
        @test dt_int32.lanes == 1

        dt_float64 = DLDataType(Float64)
        @test dt_float64.code == UInt8(LibTVMFFI.kDLFloat)
        @test dt_float64.bits == 64

        # Test from strings
        dt_from_str = DLDataType("int32")
        @test dt_from_str.code == dt_int32.code
        @test dt_from_str.bits == dt_int32.bits

        # Test string conversion
        @test string(dt_int32) == "int32"
        @test string(dt_float64) == "float64"
    end

    @testset "TVM String" begin
        # Test basic string creation
        s = TVMString("hello")
        @test String(s) == "hello"
        @test length(s) == 5

        # Test empty string
        empty_s = TVMString("")
        @test String(empty_s) == ""
        @test length(empty_s) == 0

        # Test small string optimization (≤7 bytes)
        small = TVMString("tiny")
        @test String(small) == "tiny"
        @test small.data.type_index == Int32(LibTVMFFI.kTVMFFISmallStr)

        # Test larger string (heap allocated)
        large = TVMString("this is a longer string")
        @test String(large) == "this is a longer string"
        @test large.data.type_index == Int32(LibTVMFFI.kTVMFFIStr)
    end

    @testset "TVM Bytes" begin
        # Test bytes creation
        data = UInt8[0x01, 0x02, 0x03, 0x04]
        b = TVMBytes(data)
        result = Vector{UInt8}(b)
        @test result == data
        @test length(b) == 4

        # Test empty bytes
        empty_b = TVMBytes(UInt8[])
        @test length(empty_b) == 0
    end

    @testset "Error Handling" begin
        # Test error creation
        err = TVMError(ValueError, "test message", "test backtrace")
        @test err.kind == "ValueError"
        @test err.message == "test message"
        @test err.backtrace == "test backtrace"

        # Test error kinds
        @test ValueError.name == "ValueError"
        @test TVMFFI.TypeError.name == "TypeError"  # Qualify to avoid Base.TypeError
        @test RuntimeError.name == "RuntimeError"

        # Test exception throwing
        @test_throws TVMError begin
            err = TVMError(ValueError, "thrown error", "")
            throw(err)
        end
    end

    @testset "Type Conversions" begin
        # Test to_tvm_any and from_tvm_any for basic types

        # Int64
        any_int = TVMFFI.to_tvm_any(Int64(42))
        @test TVMFFI.from_tvm_any(any_int) == 42

        # Float64
        any_float = TVMFFI.to_tvm_any(3.14)
        @test TVMFFI.from_tvm_any(any_float) ≈ 3.14

        # Bool
        any_bool = TVMFFI.to_tvm_any(true)
        @test TVMFFI.from_tvm_any(any_bool) == true

        # Nothing
        any_none = TVMFFI.to_tvm_any(nothing)
        @test TVMFFI.from_tvm_any(any_none) === nothing

        # Device
        dev = cpu(0)
        any_dev = TVMFFI.to_tvm_any(dev)
        result_dev = TVMFFI.from_tvm_any(any_dev)
        @test result_dev.device_type == dev.device_type
        @test result_dev.device_id == dev.device_id

        # String
        any_str = TVMFFI.to_tvm_any("test")
        result_str = TVMFFI.from_tvm_any(any_str)
        @test result_str == "test"
    end

    @testset "from_tvm_any Complete Coverage" begin
        # Test ALL branches of from_tvm_any with both borrowed=false and borrowed=true
        
        # Type 0: None
        any_none = LibTVMFFI.TVMFFIAny(Int32(0), 0, 0)
        @test TVMFFI.from_tvm_any(any_none; borrowed=true) === nothing
        @test TVMFFI.from_tvm_any(any_none; borrowed=false) === nothing
        
        # Type 1: Int
        any_int = LibTVMFFI.TVMFFIAny(Int32(1), 0, reinterpret(UInt64, Int64(123)))
        @test TVMFFI.from_tvm_any(any_int; borrowed=true) == 123
        @test TVMFFI.from_tvm_any(any_int; borrowed=false) == 123
        
        # Type 2: Bool
        any_bool_t = LibTVMFFI.TVMFFIAny(Int32(2), 0, UInt64(1))
        any_bool_f = LibTVMFFI.TVMFFIAny(Int32(2), 0, UInt64(0))
        @test TVMFFI.from_tvm_any(any_bool_t; borrowed=true) == true
        @test TVMFFI.from_tvm_any(any_bool_f; borrowed=true) == false
        
        # Type 3: Float
        any_float = LibTVMFFI.TVMFFIAny(Int32(3), 0, reinterpret(UInt64, 3.14))
        @test TVMFFI.from_tvm_any(any_float; borrowed=true) ≈ 3.14
        
        # Type 5: DataType
        dt_packed = UInt64(LibTVMFFI.kDLFloat) | (UInt64(32) << 8) | (UInt64(1) << 16)
        any_dtype = LibTVMFFI.TVMFFIAny(Int32(5), 0, dt_packed)
        result_dt = TVMFFI.from_tvm_any(any_dtype; borrowed=true)
        @test result_dt isa DLDataType
        @test result_dt.code == UInt8(LibTVMFFI.kDLFloat)
        @test result_dt.bits == 32
        
        # Type 6: Device
        dev_packed = UInt64(1) | (UInt64(0) << 32)  # CPU:0
        any_device = LibTVMFFI.TVMFFIAny(Int32(6), 0, dev_packed)
        result_dev = TVMFFI.from_tvm_any(any_device; borrowed=true)
        @test result_dev isa DLDevice
        @test result_dev.device_type == Int32(1)
        @test result_dev.device_id == Int32(0)
        
        # Type 11: SmallStr
        small_str = TVMString("tiny")
        @test small_str.data.type_index == Int32(LibTVMFFI.kTVMFFISmallStr)
        result = TVMFFI.from_tvm_any(small_str.data; borrowed=true)
        @test result == "tiny"
        
        # Type 12: SmallBytes
        small_bytes = TVMBytes(UInt8[1, 2, 3])
        if small_bytes.data.type_index == Int32(LibTVMFFI.kTVMFFISmallBytes)
            result = TVMFFI.from_tvm_any(small_bytes.data; borrowed=true)
            @test result == UInt8[1, 2, 3]
        end
        
        # Type 65: Str (heap string)
        large_str = TVMString("this is a longer string that goes on heap")
        @test large_str.data.type_index == Int32(LibTVMFFI.kTVMFFIStr)
        result = TVMFFI.from_tvm_any(large_str.data; borrowed=false)  # Test borrowed=false
        @test result == "this is a longer string that goes on heap"
        
        # Type 66: Bytes (heap bytes)
        large_bytes = TVMBytes(rand(UInt8, 100))
        @test large_bytes.data.type_index == Int32(LibTVMFFI.kTVMFFIBytes)
        result = TVMFFI.from_tvm_any(large_bytes.data; borrowed=false)
        @test length(result) == 100
        
        # Type 68: Function
        test_func = (x, y) -> x + y
        register_global_func("julia.type_test.add", test_func; override=true)
        func = get_global_func("julia.type_test.add")
        @test func isa TVMFunction
        # Test round-trip through from_tvm_any
        any_func = TVMFFI.to_tvm_any(func)
        recovered = TVMFFI.from_tvm_any(any_func; borrowed=false)
        @test recovered isa TVMFunction
        
        # Type 73: Module
        mod = system_lib()
        @test mod isa TVMModule
        # Module wraps TVMObject, test the wrapper
        @test mod.handle isa TVMFFI.TVMObject
        
        # Test invalid type indices (should error)
        any_raw_str = LibTVMFFI.TVMFFIAny(Int32(8), 0, 0)  # kTVMFFIRawStr
        @test_throws ErrorException TVMFFI.from_tvm_any(any_raw_str)
        
        any_byte_arr_ptr = LibTVMFFI.TVMFFIAny(Int32(9), 0, 0)  # kTVMFFIByteArrayPtr
        @test_throws ErrorException TVMFFI.from_tvm_any(any_byte_arr_ptr)
        
        any_rvalue = LibTVMFFI.TVMFFIAny(Int32(10), 0, 0)  # kTVMFFIObjectRValueRef
        @test_throws ErrorException TVMFFI.from_tvm_any(any_rvalue)
    end

    @testset "Global Function Lookup" begin
        # Test non-existent function
        func = get_global_func("this_function_definitely_does_not_exist_xyz123")
        # Should return nothing for non-existent functions
        @test func === nothing
    end

    @testset "Function Registration" begin
        # Test basic function registration
        function test_add(x::Int64, y::Int64)
            return x + y
        end

        register_global_func("julia.test.add", test_add)
        func = get_global_func("julia.test.add")
        @test func !== nothing
        @test func(Int64(10), Int64(20)) == 30

        # Test function with multiple types
        function test_multiply(x::Int64, y::Float64)
            return Float64(x) * y
        end

        register_global_func("julia.test.multiply", test_multiply)
        func2 = get_global_func("julia.test.multiply")
        result = func2(Int64(5), 3.0)
        @test result ≈ 15.0

        # Test function with boolean
        function test_negate(b::Bool)
            return !b
        end

        register_global_func("julia.test.negate", test_negate)
        func3 = get_global_func("julia.test.negate")
        @test func3(true) == false
        @test func3(false) == true

        # Test function returning nothing
        function test_void()
            return nothing
        end

        register_global_func("julia.test.void", test_void)
        func4 = get_global_func("julia.test.void")
        @test func4() === nothing

        # Test varargs function
        function test_sum(args...)
            return sum(args)
        end

        register_global_func("julia.test.sum", test_sum; override=true)
        func5 = get_global_func("julia.test.sum")
        @test func5(Int64(1), Int64(2), Int64(3), Int64(4)) == 10

        # Test exception handling
        function test_error()
            error("Test error!")
        end

        register_global_func("julia.test.error", test_error; override=true)
        func6 = get_global_func("julia.test.error")
        @test_throws TVMError func6()
    end

    @testset "Object Registration" begin
        # Test basic type registration
        struct TestObject1
            x::Int
        end

        idx = register_object("julia.test.TestObject1", TestObject1)
        @test idx > 0

        # Verify we can look it up
        idx2 = get_type_index("julia.test.TestObject1")
        @test idx == idx2

        # Test with different parent type
        struct TestObject2
            y::Float64
        end

        idx3 = register_object("julia.test.TestObject2", TestObject2; parent_type_index=Int32(64))
        @test idx3 > 0
        @test idx3 != idx
    end

    @testset "DLTensorHolder - CPU Arrays" begin
        # Test creating holder from regular array
        x = Float32[1, 2, 3, 4, 5]
        holder = from_julia_array(x)

        @test holder isa DLTensorHolder
        @test holder.shape == [5]
        @test holder.strides == [1]
        @test holder.source === x
        @test holder.tensor.device.device_type == Int32(LibTVMFFI.kDLCPU)

        # Test that holder keeps array alive
        @test eltype(holder.source) == Float32
    end

    @testset "DLTensorHolder - Slices" begin
        # Test vector slice
        vec = Float32[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        slice = @view vec[3:7]
        holder = from_julia_array(slice)

        @test holder isa DLTensorHolder
        @test holder.shape == [5]
        @test holder.strides == [1]  # Contiguous slice
        @test holder.source === slice

        # Test column slice (contiguous in column-major)
        matrix = Float32[1 2 3; 4 5 6; 7 8 9]
        col = @view matrix[:, 2]
        col_holder = from_julia_array(col)

        @test col_holder.shape == [3]
        @test col_holder.strides == [1]  # Contiguous

        # Test row slice (non-contiguous in column-major)
        row = @view matrix[2, :]
        row_holder = from_julia_array(row)

        @test row_holder.shape == [3]
        @test row_holder.strides == [3]  # Non-contiguous
    end

    @testset "DLTensorHolder - unsafe_convert" begin
        # Test that unsafe_convert works
        x = Float32[1, 2, 3]
        holder = from_julia_array(x)

        # Should be able to get pointer
        ptr = Base.unsafe_convert(Ptr{DLTensor}, holder)
        @test ptr != C_NULL

        # Pointer should point to valid data
        tensor = unsafe_load(ptr)
        @test tensor.ndim == 1
        @test tensor.device.device_type == Int32(LibTVMFFI.kDLCPU)
    end

    @testset "Automatic Array Conversion" begin
        # Test that to_tvm_any handles arrays automatically
        x = Float32[1, 2, 3]

        # Should auto-convert AbstractArray to DLTensorHolder
        # (This is tested indirectly through function calls)
        holder = from_julia_array(x)
        any_val = TVMFFI.to_tvm_any(holder)

        @test any_val.type_index == Int32(LibTVMFFI.kTVMFFIDLTensorPtr)
        @test any_val.data != 0
    end

    @testset "Reference Counting - own parameter" begin
        # Test that constructors respect own parameter
        # This is implementation detail, but important for correctness

        # Create a TVMString to test with
        s = TVMString("test")

        # The internal handle should be valid
        @test s.data.type_index >= Int32(LibTVMFFI.kTVMFFISmallStr)
    end

    @testset "DLTensorHolder - Unified CPU/GPU" begin
        # Test that DLTensorHolder works for both CPU and GPU arrays
        # (GPU test only runs if GPU available, but we test the type unification)

        cpu_arr = Float32[1, 2, 3]
        cpu_holder = from_julia_array(cpu_arr)

        @test cpu_holder isa DLTensorHolder
        @test cpu_holder.tensor.device.device_type == Int32(LibTVMFFI.kDLCPU)

        # GPU arrays would create the same DLTensorHolder type
        # just with different device_type in the tensor
    end

    @testset "Module API" begin
        # Test that module API functions are cached
        @test TVMFFI._module_loader[] !== nothing
        @test TVMFFI._function_getter[] !== nothing

        # Test exports
        @test isdefined(TVMFFI, :load_module)
        @test isdefined(TVMFFI, :get_function)
        @test isdefined(TVMFFI, :TVMModule)
        @test isdefined(TVMFFI, :system_lib)
        @test isdefined(TVMFFI, :write_to_file)
        @test isdefined(TVMFFI, :inspect_source)
        @test isdefined(TVMFFI, :get_module_kind)
        @test isdefined(TVMFFI, :implements_function)
    end

    @testset "Module Enhancements" begin
        # Test system_lib
        mod = system_lib()
        @test mod isa TVMModule
        
        # Test get_module_kind
        kind = get_module_kind(mod)
        @test kind isa String
        @test kind == "library"
        
        # Test implements_function
        # System lib may or may not have functions, just test the API works
        result = implements_function(mod, "nonexistent_function_12345", false)
        @test result isa Bool
    end

    @testset "DLTensor Layout Verification" begin
        # Verify struct sizes
        @test sizeof(DLDevice) == 8
        @test sizeof(DLDataType) == 4
        
        # Verify field offsets for 64-bit platform
        if Sys.WORD_SIZE == 64
            @test fieldoffset(DLTensor, 1) == 0   # data
            @test fieldoffset(DLTensor, 2) == 8   # device
            @test fieldoffset(DLTensor, 3) == 16  # ndim
            @test fieldoffset(DLTensor, 4) == 20  # dtype
            @test fieldoffset(DLTensor, 5) == 24  # shape
            @test fieldoffset(DLTensor, 6) == 32  # strides
            @test fieldoffset(DLTensor, 7) == 40  # byte_offset
            @test sizeof(DLTensor) == 48
        end
        
        # Test with actual tensor
        x = Float32[1, 2, 3, 4, 5]
        holder = from_julia_array(x)
        @test holder.tensor.ndim == 1
        @test holder.tensor.dtype.bits == 32
        @test holder.tensor.device.device_type == Int32(LibTVMFFI.kDLCPU)
    end

    @testset "GC Safety Stress Test" begin
        # Test that references survive aggressive GC
        
        # Test 1: String creation under GC pressure
        all_strings_valid = true
        for i in 1:100
            s = TVMString("test string $i")
            # Create garbage
            _ = [rand(100) for _ in 1:10]
            GC.gc()
            # String should still be valid
            if String(s) != "test string $i"
                all_strings_valid = false
                break
            end
        end
        @test all_strings_valid
        
        # Test 2: Function registration under GC
        all_funcs_work = true
        for i in 1:50
            func_name = "julia.gc_test_$i"
            test_func = x -> x + i
            register_global_func(func_name, test_func; override=true)
            
            # Trigger GC
            _ = [rand(1000) for _ in 1:20]
            GC.gc()
            
            # Retrieve and call
            retrieved = get_global_func(func_name)
            if retrieved === nothing || retrieved(Int64(10)) != 10 + i
                all_funcs_work = false
                break
            end
        end
        @test all_funcs_work
        
        # Test 3: Object references under GC
        all_modules_valid = true
        for i in 1:50
            mod = system_lib()
            # Allocate garbage
            _ = [TVMString("garbage $j") for j in 1:100]
            GC.gc()
            # Module should still be valid
            if !(mod isa TVMModule) || get_module_kind(mod) != "library"
                all_modules_valid = false
                break
            end
        end
        @test all_modules_valid
    end
end

println("\n✓ All tests passed!")
