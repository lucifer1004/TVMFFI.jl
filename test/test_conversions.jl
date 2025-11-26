# Tests for Type Conversions (TVMAny / take_value / copy_value)

@testset "TVMAny Constructors" begin
    # Test TVMAny(value) constructors for basic types

    # Int64
    any_int = TVMAny(Int64(42))
    @test take_value(any_int) == 42

    # Float64
    any_float = TVMAny(3.14)
    @test take_value(any_float) ≈ 3.14

    # Bool
    any_bool = TVMAny(true)
    @test take_value(any_bool) == true

    # Nothing
    any_none = TVMAny(nothing)
    @test take_value(any_none) === nothing

    # Device
    dev = cpu(0)
    any_dev = TVMAny(dev)
    result_dev = take_value(any_dev)
    @test result_dev.device_type == dev.device_type
    @test result_dev.device_id == dev.device_id

    # DataType
    dt = DLDataType(UInt8(LibTVMFFI.kDLFloat), UInt8(32), UInt16(1))
    any_dt = TVMAny(dt)
    result_dt = take_value(any_dt)
    @test result_dt.code == dt.code
    @test result_dt.bits == dt.bits

    # String
    any_str = TVMAny("test")
    result_str = take_value(any_str)
    @test result_str == "test"
end

@testset "take_value Complete Coverage" begin
    # Test ALL type branches with take_value (owned semantics)

    # Type 0: None
    any_none = LibTVMFFI.TVMFFIAny(Int32(0), 0, 0)
    @test take_value(TVMAny(any_none)) === nothing

    # Type 1: Int
    any_int = LibTVMFFI.TVMFFIAny(Int32(1), 0, reinterpret(UInt64, Int64(123)))
    @test take_value(TVMAny(any_int)) == 123

    # Type 2: Bool
    any_bool_t = LibTVMFFI.TVMFFIAny(Int32(2), 0, UInt64(1))
    any_bool_f = LibTVMFFI.TVMFFIAny(Int32(2), 0, UInt64(0))
    @test take_value(TVMAny(any_bool_t)) == true
    @test take_value(TVMAny(any_bool_f)) == false

    # Type 3: Float
    any_float = LibTVMFFI.TVMFFIAny(Int32(3), 0, reinterpret(UInt64, 3.14))
    @test take_value(TVMAny(any_float)) ≈ 3.14

    # Type 5: DataType
    dt_packed = UInt64(LibTVMFFI.kDLFloat) | (UInt64(32) << 8) | (UInt64(1) << 16)
    any_dtype = LibTVMFFI.TVMFFIAny(Int32(5), 0, dt_packed)
    result_dt = take_value(TVMAny(any_dtype))
    @test result_dt isa DLDataType
    @test result_dt.code == UInt8(LibTVMFFI.kDLFloat)
    @test result_dt.bits == 32

    # Type 6: Device
    dev_packed = UInt64(1) | (UInt64(0) << 32)  # CPU:0
    any_device = LibTVMFFI.TVMFFIAny(Int32(6), 0, dev_packed)
    result_dev = take_value(TVMAny(any_device))
    @test result_dev isa DLDevice
    @test result_dev.device_type == Int32(1)
    @test result_dev.device_id == Int32(0)
end

@testset "copy_value Complete Coverage" begin
    # Test copy_value (borrowed semantics via TVMFFIAnyViewToOwnedAny)

    # Type 0: None
    any_none = LibTVMFFI.TVMFFIAny(Int32(0), 0, 0)
    @test copy_value(TVMAnyView(any_none)) === nothing

    # Type 1: Int
    any_int = LibTVMFFI.TVMFFIAny(Int32(1), 0, reinterpret(UInt64, Int64(456)))
    @test copy_value(TVMAnyView(any_int)) == 456

    # Type 2: Bool
    any_bool = LibTVMFFI.TVMFFIAny(Int32(2), 0, UInt64(1))
    @test copy_value(TVMAnyView(any_bool)) == true

    # Type 3: Float
    any_float = LibTVMFFI.TVMFFIAny(Int32(3), 0, reinterpret(UInt64, 2.71))
    @test copy_value(TVMAnyView(any_float)) ≈ 2.71

    # Type 5: DataType
    dt_packed = UInt64(LibTVMFFI.kDLFloat) | (UInt64(64) << 8) | (UInt64(1) << 16)
    any_dtype = LibTVMFFI.TVMFFIAny(Int32(5), 0, dt_packed)
    result_dt = copy_value(TVMAnyView(any_dtype))
    @test result_dt isa DLDataType
    @test result_dt.bits == 64

    # Type 6: Device
    dev_packed = UInt64(2) | (UInt64(1) << 32)  # CUDA:1
    any_device = LibTVMFFI.TVMFFIAny(Int32(6), 0, dev_packed)
    result_dev = copy_value(TVMAnyView(any_device))
    @test result_dev isa DLDevice
    @test result_dev.device_type == Int32(2)
    @test result_dev.device_id == Int32(1)

    # Type 11: SmallStr (via TVMString which creates proper Any)
    small_str = TVMString("tiny")
    @test small_str.data.type_index == Int32(LibTVMFFI.kTVMFFISmallStr)
    result = copy_value(TVMAnyView(small_str.data))
    @test result == "tiny"

    # Type 12: SmallBytes
    small_bytes = TVMBytes(UInt8[1, 2, 3])
    if small_bytes.data.type_index == Int32(LibTVMFFI.kTVMFFISmallBytes)
        result = copy_value(TVMAnyView(small_bytes.data))
        @test result == UInt8[1, 2, 3]
    end

    # Type 65: Str (heap string)
    large_str = TVMString("this is a longer string that goes on heap")
    @test large_str.data.type_index == Int32(LibTVMFFI.kTVMFFIStr)
    result = copy_value(TVMAnyView(large_str.data))
    @test result == "this is a longer string that goes on heap"

    # Type 66: Bytes (heap bytes)
    large_bytes = TVMBytes(rand(UInt8, 100))
    @test large_bytes.data.type_index == Int32(LibTVMFFI.kTVMFFIBytes)
    result = copy_value(TVMAnyView(large_bytes.data))
    @test length(result) == 100

    # Type 68: Function
    test_func = (x, y) -> x + y
    register_global_func("julia.type_test.add", test_func; override = true)
    func = get_global_func("julia.type_test.add")
    @test func isa TVMFunction
    # Test round-trip through copy_value
    func_any = TVMAny(func)
    recovered = copy_value(TVMAnyView(raw_data(func_any)))
    @test recovered isa TVMFunction

    # Type 73: Module
    mod = system_lib()
    @test mod isa TVMModule
    # Module wraps TVMObject, test the wrapper
    @test mod.handle isa TVMFFI.TVMObject
end

@testset "TVMAny and TVMAnyView Types" begin
    # Test TVMAnyView for POD types (no refcounting)
    raw_int = LibTVMFFI.TVMFFIAny(Int32(1), 0, reinterpret(UInt64, Int64(42)))
    view = TVMAnyView(raw_int)
    @test TVMFFI.type_index(view) == Int32(1)
    @test TVMFFI.is_object(view) == false
    @test TVMFFI.is_none(view) == false
    @test copy_value(view) == 42

    # Test TVMAnyView for None
    raw_none = LibTVMFFI.TVMFFIAny(Int32(0), 0, 0)
    view_none = TVMAnyView(raw_none)
    @test TVMFFI.is_none(view_none) == true
    @test copy_value(view_none) === nothing

    # Test TVMAny for POD types (no finalizer needed)
    any_int = TVMAny(LibTVMFFI.TVMFFIAny(Int32(1), 0, reinterpret(UInt64, Int64(123))))
    @test TVMFFI.type_index(any_int) == Int32(1)
    @test TVMFFI.is_object(any_int) == false
    @test take_value(any_int) == 123

    # Test TVMAny for Float
    any_float = TVMAny(LibTVMFFI.TVMFFIAny(Int32(3), 0, reinterpret(UInt64, 3.14)))
    @test take_value(any_float) ≈ 3.14

    # Test TVMAny for Bool
    any_bool = TVMAny(LibTVMFFI.TVMFFIAny(Int32(2), 0, UInt64(1)))
    @test take_value(any_bool) == true

    # Test TVMAny for Device
    dev_packed = UInt64(1) | (UInt64(2) << 32)  # CPU:2
    any_device = TVMAny(LibTVMFFI.TVMFFIAny(Int32(6), 0, dev_packed))
    result_dev = take_value(any_device)
    @test result_dev isa DLDevice
    @test result_dev.device_type == Int32(1)
    @test result_dev.device_id == Int32(2)

    # Test TVMAny for DataType
    dt_packed = UInt64(LibTVMFFI.kDLFloat) | (UInt64(64) << 8) | (UInt64(1) << 16)
    any_dtype = TVMAny(LibTVMFFI.TVMFFIAny(Int32(5), 0, dt_packed))
    result_dt = take_value(any_dtype)
    @test result_dt isa DLDataType
    @test result_dt.code == UInt8(LibTVMFFI.kDLFloat)
    @test result_dt.bits == 64
end

@testset "TVMAny/TVMAnyView with Objects" begin
    # Test with real TVM objects - String
    str = TVMString("hello world from TVM")
    @test str.data.type_index == Int32(LibTVMFFI.kTVMFFIStr)  # heap string

    # Create view and copy
    view = TVMAnyView(str.data)
    @test TVMFFI.is_object(view) == true
    copied = copy_value(view)
    @test copied == "hello world from TVM"

    # Test with Function object
    test_fn = x -> x * 2
    register_global_func("julia.test.any_view_fn", test_fn; override = true)
    func = get_global_func("julia.test.any_view_fn")
    @test func isa TVMFunction

    # Create Any from function and take value
    func_any = TVMAny(func)
    @test TVMFFI.is_object(func_any) == true
    recovered = take_value(func_any)
    @test recovered isa TVMFunction

    # Test view → owned conversion via TVMAny(view)
    str2 = TVMString("test view to owned")
    view2 = TVMAnyView(str2.data)
    owned = TVMAny(view2)  # Uses TVMFFIAnyViewToOwnedAny
    @test TVMFFI.is_object(owned) == true
    result = take_value(owned)
    @test result == "test view to owned"
end

@testset "TVMAny with DLTensorHolder" begin
    # Test TVMAny with DLTensorHolder
    arr = Float32[1.0, 2.0, 3.0, 4.0]
    holder = from_julia_array(arr)
    any = TVMAny(holder)
    @test TVMFFI.type_index(any) == Int32(LibTVMFFI.kTVMFFIDLTensorPtr)
    
    # Note: can't use take_value for DLTensorPtr since we don't own the data
    # The holder owns it
end

@testset "TVMAnyView Display" begin
    # Test show method
    view = TVMAnyView(LibTVMFFI.TVMFFIAny(Int32(1), 0, UInt64(42)))
    @test occursin("TVMAnyView", repr(view))
    @test occursin("type_index=1", repr(view))

    any = TVMAny(LibTVMFFI.TVMFFIAny(Int32(3), 0, reinterpret(UInt64, 2.71)))
    @test occursin("TVMAny", repr(any))
    @test occursin("type_index=3", repr(any))
end
