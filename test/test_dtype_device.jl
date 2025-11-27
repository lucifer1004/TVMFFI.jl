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

    # Test all device types (just type/id, not actual hardware)
    cuda_dev = cuda(0)
    @test cuda_dev.device_type == Int32(LibTVMFFI.kDLCUDA)
    @test cuda_dev.device_id == 0

    cuda_dev1 = cuda(1)
    @test cuda_dev1.device_id == 1

    opencl_dev = opencl(0)
    @test opencl_dev.device_type == Int32(LibTVMFFI.kDLOpenCL)
    @test opencl_dev.device_id == 0

    vulkan_dev = vulkan(0)
    @test vulkan_dev.device_type == Int32(LibTVMFFI.kDLVulkan)
    @test vulkan_dev.device_id == 0

    metal_dev = metal(0)
    @test metal_dev.device_type == Int32(LibTVMFFI.kDLMetal)
    @test metal_dev.device_id == 0

    rocm_dev = rocm(0)
    @test rocm_dev.device_type == Int32(LibTVMFFI.kDLROCM)
    @test rocm_dev.device_id == 0

    # Test with non-zero device IDs
    @test vulkan(2).device_id == 2
    @test opencl(3).device_id == 3
end

@testset "Device from DLDeviceType" begin
    # Test DLDevice constructor with DLDeviceType
    dev = DLDevice(LibTVMFFI.kDLCPU, 0)
    @test dev.device_type == Int32(LibTVMFFI.kDLCPU)
    @test dev.device_id == 0

    dev2 = DLDevice(LibTVMFFI.kDLCUDA, 2)
    @test dev2.device_type == Int32(LibTVMFFI.kDLCUDA)
    @test dev2.device_id == 2
end

@testset "Device show()" begin
    # Test pretty printing
    cpu_str = sprint(show, cpu(0))
    @test occursin("CPU", cpu_str)
    @test occursin("0", cpu_str)

    cuda_str = sprint(show, cuda(1))
    @test occursin("CUDA", cuda_str)
    @test occursin("1", cuda_str)

    opencl_str = sprint(show, opencl(0))
    @test occursin("OpenCL", opencl_str)

    vulkan_str = sprint(show, vulkan(0))
    @test occursin("Vulkan", vulkan_str)

    metal_str = sprint(show, metal(0))
    @test occursin("Metal", metal_str)

    rocm_str = sprint(show, rocm(0))
    @test occursin("ROCm", rocm_str)

    # Test unknown device type (VPI = 8)
    vpi_dev = DLDevice(Int32(LibTVMFFI.kDLVPI), 0)
    vpi_str = sprint(show, vpi_dev)
    @test occursin("VPI", vpi_str)
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
