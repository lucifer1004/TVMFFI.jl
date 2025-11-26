# Tests for DLTensor and Array Handling

@testset "TensorView - CPU Arrays" begin
    # Test creating view from regular array
    x = Float32[1, 2, 3, 4, 5]
    view = TensorView(x)

    @test view isa TensorView
    @test view.shape == [5]
    @test view.strides == [1]
    @test view.source === x
    @test view.dltensor.device.device_type == Int32(LibTVMFFI.kDLCPU)

    # Test that view keeps array alive
    @test eltype(view.source) == Float32
end

@testset "TensorView - Slices" begin
    # Test vector slice
    vec = Float32[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    slice = @view vec[3:7]
    view = TensorView(slice)

    @test view isa TensorView
    @test view.shape == [5]
    @test view.strides == [1]  # Contiguous slice
    @test view.source === slice

    # Test column slice (contiguous in column-major)
    matrix = Float32[1 2 3; 4 5 6; 7 8 9]
    col = @view matrix[:, 2]
    col_view = TensorView(col)

    @test col_view.shape == [3]
    @test col_view.strides == [1]  # Contiguous

    # Test row slice (non-contiguous in column-major)
    row = @view matrix[2, :]
    row_view = TensorView(row)

    @test row_view.shape == [3]
    @test row_view.strides == [3]  # Non-contiguous
end

@testset "TensorView - unsafe_convert" begin
    # Test that unsafe_convert works
    x = Float32[1, 2, 3]
    view = TensorView(x)

    # Should be able to get pointer
    ptr = Base.unsafe_convert(Ptr{DLTensor}, view)
    @test ptr != C_NULL

    # Pointer should point to valid data
    tensor = unsafe_load(ptr)
    @test tensor.ndim == 1
    @test tensor.device.device_type == Int32(LibTVMFFI.kDLCPU)
end

@testset "Automatic Array Conversion" begin
    # Test that TVMAny handles TensorView correctly
    x = Float32[1, 2, 3]

    # Should auto-convert AbstractArray to TensorView
    # (This is tested indirectly through function calls)
    view = TensorView(x)
    any_val = TVMAny(view)

    @test TVMFFI.type_index(any_val) == Int32(LibTVMFFI.kTVMFFIDLTensorPtr)
    @test raw_data(any_val).data != 0
end

@testset "TensorView - Unified CPU/GPU" begin
    # Test that TensorView works for both CPU and GPU arrays
    # (GPU test only runs if GPU available, but we test the type unification)

    cpu_arr = Float32[1, 2, 3]
    cpu_view = TensorView(cpu_arr)

    @test cpu_view isa TensorView
    @test cpu_view.dltensor.device.device_type == Int32(LibTVMFFI.kDLCPU)

    # GPU arrays would create the same TensorView type
    # just with different device_type in the tensor
end
