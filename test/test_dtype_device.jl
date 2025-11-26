# Tests for Device and DataType APIs

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

    # Test dtype_to_julia_type conversion
    @test dtype_to_julia_type(DLDataType(Float16)) == Float16
    @test dtype_to_julia_type(DLDataType(Float32)) == Float32
    @test dtype_to_julia_type(DLDataType(Float64)) == Float64
    @test dtype_to_julia_type(DLDataType(Int8)) == Int8
    @test dtype_to_julia_type(DLDataType(Int16)) == Int16
    @test dtype_to_julia_type(DLDataType(Int32)) == Int32
    @test dtype_to_julia_type(DLDataType(Int64)) == Int64
    @test dtype_to_julia_type(DLDataType(UInt8)) == UInt8
    @test dtype_to_julia_type(DLDataType(UInt16)) == UInt16
    @test dtype_to_julia_type(DLDataType(UInt32)) == UInt32
    @test dtype_to_julia_type(DLDataType(UInt64)) == UInt64
    @test dtype_to_julia_type(DLDataType(Bool)) == Bool
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
    view = TensorView(x)
    @test view.dltensor.ndim == 1
    @test view.dltensor.dtype.bits == 32
    @test view.dltensor.device.device_type == Int32(LibTVMFFI.kDLCPU)
end
