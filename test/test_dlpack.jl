# Tests for DLPack zero-copy tensor exchange

@testset "DLPack Zero-Copy" begin
    @testset "TVMTensor(arr) - Basic" begin
        # Test basic conversion
        arr = Float32[1, 2, 3, 4, 5]
        tensor = TVMTensor(arr)

        @test tensor isa TVMTensor
        @test size(tensor) == (5,)
        @test dtype(tensor).bits == 32
        @test device(tensor).device_type == Int32(LibTVMFFI.kDLCPU)
    end

    @testset "TVMTensor(arr) - Multi-dimensional" begin
        # Test 2D array
        arr = rand(Float32, 3, 4)
        tensor = TVMTensor(arr)

        @test tensor isa TVMTensor
        # Note: DLPack reverses dimensions (row-major vs col-major)
        # TVM sees (4, 3) instead of (3, 4)
        @test ndims(tensor) == 2
    end

    @testset "TVMTensor(arr) - Various dtypes" begin
        # Test different data types
        for T in [Float32, Float64, Int32, Int64]
            arr = T[1, 2, 3]
            tensor = TVMTensor(arr)
            @test tensor isa TVMTensor
        end
    end

    @testset "from_dlpack - Basic" begin
        # Create a TVM tensor and convert back
        arr = Float32[1, 2, 3, 4, 5]
        tensor = TVMTensor(arr)

        # Convert back to Julia array
        arr2 = from_dlpack(tensor)

        @test arr2 isa AbstractArray
        @test length(arr2) == length(arr)
    end

    @testset "Zero-copy verification" begin
        # Verify that modifications through one view affect the other
        arr = Float32[1, 2, 3, 4, 5]
        tensor = TVMTensor(arr)

        # Modify original array
        arr[1] = 99.0f0

        # Check if tensor sees the change
        # This requires getting the data from the tensor
        arr2 = from_dlpack(tensor)

        # Note: Due to dimension reversal, we check by value
        @test 99.0f0 in arr2
    end

    @testset "TVMAny with TVMTensor" begin
        # Test TVMAny creation using TVMTensor (DLPack-based)
        arr = Float32[1, 2, 3]
        tensor = TVMTensor(arr)
        any = TVMFFI.TVMAny(tensor)

        @test TVMFFI.type_index(any) == Int32(LibTVMFFI.kTVMFFITensor)
    end
end

@testset "TVMTensor Methods" begin
    @testset "shape and size" begin
        arr = rand(Float32, 3, 4, 5)
        tensor = TVMTensor(arr)

        @test shape(tensor) isa Vector{Int64}
        @test length(shape(tensor)) == 3
        @test size(tensor) isa Tuple
        @test ndims(tensor) == 3
        @test length(tensor) == 60

        # size(tensor, dim)
        @test size(tensor, 1) > 0
        @test size(tensor, 2) > 0
        @test size(tensor, 3) > 0
        @test size(tensor, 4) == 1  # Out of bounds returns 1
    end

    @testset "dtype and device" begin
        for T in [Float32, Float64, Int32, Int64]
            arr = T[1, 2, 3]
            tensor = TVMTensor(arr)
            @test dtype(tensor) isa DLDataType
            @test device(tensor) isa DLDevice
            @test device(tensor).device_type == Int32(LibTVMFFI.kDLCPU)
        end
    end

    @testset "strides and contiguity" begin
        arr = rand(Float32, 3, 4)
        tensor = TVMTensor(arr)

        str = TVMFFI.strides(tensor)
        @test str isa Vector{Int64}
        @test length(str) == 2

        # Julia arrays are F-contiguous (column-major), TVMTensor.is_contiguous
        # checks C-contiguous (row-major), so a Julia array won't be C-contiguous
        # This is expected behavior - the tensor preserves Julia's memory layout
        @test str == [1, 3]  # F-contiguous strides for 3x4 array
    end

    @testset "data_ptr" begin
        arr = Float32[1, 2, 3]
        tensor = TVMTensor(arr)

        ptr = TVMFFI.data_ptr(tensor)
        @test ptr isa Ptr{Cvoid}
        @test ptr != C_NULL
    end

    @testset "show and summary" begin
        arr = Float32[1, 2, 3]
        tensor = TVMTensor(arr)

        # Test show
        str = sprint(show, tensor)
        @test occursin("TVMTensor", str)
        @test occursin("float32", str)

        # Test summary
        summary_str = sprint(Base.summary, tensor)
        @test occursin("TVMTensor", summary_str)
    end
end

@testset "TensorView from TVMTensor" begin
    arr = rand(Float32, 3, 4)
    tensor = TVMTensor(arr)

    # Create TensorView from TVMTensor
    view = TensorView(tensor)

    @test view isa TensorView
    @test TVMFFI.is_tvm_owned(view) == true
    @test TVMFFI.is_julia_owned(view) == false
    @test view.source === tensor
end

@testset "DLManagedTensor" begin
    # Test to_dlmanaged_tensor
    arr = Float32[1, 2, 3]
    view = TensorView(arr)

    dlm_ptr = TVMFFI.to_dlmanaged_tensor(view)
    @test dlm_ptr isa Ptr{TVMFFI.DLManagedTensor}
    @test dlm_ptr != C_NULL

    # Load and verify
    dlm = unsafe_load(dlm_ptr)
    @test dlm.dl_tensor.ndim == 1
    @test dlm.deleter != C_NULL  # Has deleter function
end

@testset "Device Type Names" begin
    # Test _device_type_to_name helper (GPU types only, CPU uses different path)
    @test TVMFFI._device_type_to_name(Int32(LibTVMFFI.kDLCUDA)) == "CUDA"
    @test TVMFFI._device_type_to_name(Int32(LibTVMFFI.kDLROCM)) == "ROCm"  # Note: ROCM not ROCm
    @test TVMFFI._device_type_to_name(Int32(LibTVMFFI.kDLMetal)) == "Metal"
    @test TVMFFI._device_type_to_name(Int32(LibTVMFFI.kDLVulkan)) == "Vulkan"
    @test TVMFFI._device_type_to_name(Int32(LibTVMFFI.kDLOpenCL)) == "OpenCL"
    @test TVMFFI._device_type_to_name(Int32(LibTVMFFI.kDLOneAPI)) == "oneAPI"
    @test TVMFFI._device_type_to_name(Int32(LibTVMFFI.kDLCUDAHost)) == "CUDA Host"
    @test TVMFFI._device_type_to_name(Int32(LibTVMFFI.kDLCUDAManaged)) == "CUDA Managed"

    # CPU is not in the GPU-focused mapping, returns Unknown
    cpu_name = TVMFFI._device_type_to_name(Int32(LibTVMFFI.kDLCPU))
    @test occursin("Unknown", cpu_name)

    # Unknown type
    unknown = TVMFFI._device_type_to_name(Int32(9999))
    @test occursin("Unknown", unknown)
end
