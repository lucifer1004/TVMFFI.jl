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
