# Tests for TVM String and Bytes

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

@testset "TVMString Type System" begin
    # Test type_index and type_key for TVMString
    @test type_index(TVMString) == Int32(LibTVMFFI.kTVMFFIStr)  # Canonical type
    @test type_key(TVMString) == "ffi.Str"

    # Test instance type_index (small string)
    small = TVMString("tiny")
    @test type_index(small) == Int32(LibTVMFFI.kTVMFFISmallStr)

    # Test instance type_index (large string)
    large = TVMString("this is a longer string that won't fit inline")
    @test type_index(large) == Int32(LibTVMFFI.kTVMFFIStr)
end

@testset "TVMBytes Type System" begin
    # Test type_index and type_key for TVMBytes
    @test type_index(TVMBytes) == Int32(LibTVMFFI.kTVMFFIBytes)  # Canonical type
    @test type_key(TVMBytes) == "ffi.Bytes"

    # Test instance type_index (small bytes)
    small = TVMBytes(UInt8[1, 2, 3])
    @test type_index(small) == Int32(LibTVMFFI.kTVMFFISmallBytes)

    # Test instance type_index (large bytes)
    large = TVMBytes(collect(UInt8, 1:100))
    @test type_index(large) == Int32(LibTVMFFI.kTVMFFIBytes)
end
