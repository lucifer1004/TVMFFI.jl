# Tests for Type Conversions (to_tvm_any / from_tvm_any)

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
    @test TVMFFI.from_tvm_any(any_none; borrowed = true) === nothing
    @test TVMFFI.from_tvm_any(any_none; borrowed = false) === nothing

    # Type 1: Int
    any_int = LibTVMFFI.TVMFFIAny(Int32(1), 0, reinterpret(UInt64, Int64(123)))
    @test TVMFFI.from_tvm_any(any_int; borrowed = true) == 123
    @test TVMFFI.from_tvm_any(any_int; borrowed = false) == 123

    # Type 2: Bool
    any_bool_t = LibTVMFFI.TVMFFIAny(Int32(2), 0, UInt64(1))
    any_bool_f = LibTVMFFI.TVMFFIAny(Int32(2), 0, UInt64(0))
    @test TVMFFI.from_tvm_any(any_bool_t; borrowed = true) == true
    @test TVMFFI.from_tvm_any(any_bool_f; borrowed = true) == false

    # Type 3: Float
    any_float = LibTVMFFI.TVMFFIAny(Int32(3), 0, reinterpret(UInt64, 3.14))
    @test TVMFFI.from_tvm_any(any_float; borrowed = true) ≈ 3.14

    # Type 5: DataType
    dt_packed = UInt64(LibTVMFFI.kDLFloat) | (UInt64(32) << 8) | (UInt64(1) << 16)
    any_dtype = LibTVMFFI.TVMFFIAny(Int32(5), 0, dt_packed)
    result_dt = TVMFFI.from_tvm_any(any_dtype; borrowed = true)
    @test result_dt isa DLDataType
    @test result_dt.code == UInt8(LibTVMFFI.kDLFloat)
    @test result_dt.bits == 32

    # Type 6: Device
    dev_packed = UInt64(1) | (UInt64(0) << 32)  # CPU:0
    any_device = LibTVMFFI.TVMFFIAny(Int32(6), 0, dev_packed)
    result_dev = TVMFFI.from_tvm_any(any_device; borrowed = true)
    @test result_dev isa DLDevice
    @test result_dev.device_type == Int32(1)
    @test result_dev.device_id == Int32(0)

    # Type 11: SmallStr
    small_str = TVMString("tiny")
    @test small_str.data.type_index == Int32(LibTVMFFI.kTVMFFISmallStr)
    result = TVMFFI.from_tvm_any(small_str.data; borrowed = true)
    @test result == "tiny"

    # Type 12: SmallBytes
    small_bytes = TVMBytes(UInt8[1, 2, 3])
    if small_bytes.data.type_index == Int32(LibTVMFFI.kTVMFFISmallBytes)
        result = TVMFFI.from_tvm_any(small_bytes.data; borrowed = true)
        @test result == UInt8[1, 2, 3]
    end

    # Type 65: Str (heap string)
    large_str = TVMString("this is a longer string that goes on heap")
    @test large_str.data.type_index == Int32(LibTVMFFI.kTVMFFIStr)
    result = TVMFFI.from_tvm_any(large_str.data; borrowed = true)
    @test result == "this is a longer string that goes on heap"

    # Type 66: Bytes (heap bytes)
    large_bytes = TVMBytes(rand(UInt8, 100))
    @test large_bytes.data.type_index == Int32(LibTVMFFI.kTVMFFIBytes)
    result = TVMFFI.from_tvm_any(large_bytes.data; borrowed = true)
    @test length(result) == 100

    # Type 68: Function
    test_func = (x, y) -> x + y
    register_global_func("julia.type_test.add", test_func; override = true)
    func = get_global_func("julia.type_test.add")
    @test func isa TVMFunction
    # Test round-trip through from_tvm_any
    any_func = TVMFFI.to_tvm_any(func)
    recovered = TVMFFI.from_tvm_any(any_func; borrowed = true)
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
