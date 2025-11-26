# Tests for DLTensor and Array Handling

@testset "DLTensorHolder - CPU Arrays" begin
    # Test creating holder from regular array
    x = Float32[1, 2, 3, 4, 5]
    holder = DLTensorHolder(x)

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
    holder = DLTensorHolder(slice)

    @test holder isa DLTensorHolder
    @test holder.shape == [5]
    @test holder.strides == [1]  # Contiguous slice
    @test holder.source === slice

    # Test column slice (contiguous in column-major)
    matrix = Float32[1 2 3; 4 5 6; 7 8 9]
    col = @view matrix[:, 2]
    col_holder = DLTensorHolder(col)

    @test col_holder.shape == [3]
    @test col_holder.strides == [1]  # Contiguous

    # Test row slice (non-contiguous in column-major)
    row = @view matrix[2, :]
    row_holder = DLTensorHolder(row)

    @test row_holder.shape == [3]
    @test row_holder.strides == [3]  # Non-contiguous
end

@testset "DLTensorHolder - unsafe_convert" begin
    # Test that unsafe_convert works
    x = Float32[1, 2, 3]
    holder = DLTensorHolder(x)

    # Should be able to get pointer
    ptr = Base.unsafe_convert(Ptr{DLTensor}, holder)
    @test ptr != C_NULL

    # Pointer should point to valid data
    tensor = unsafe_load(ptr)
    @test tensor.ndim == 1
    @test tensor.device.device_type == Int32(LibTVMFFI.kDLCPU)
end

@testset "Automatic Array Conversion" begin
    # Test that TVMAny handles DLTensorHolder correctly
    x = Float32[1, 2, 3]

    # Should auto-convert AbstractArray to DLTensorHolder
    # (This is tested indirectly through function calls)
    holder = DLTensorHolder(x)
    any_val = TVMAny(holder)

    @test TVMFFI.type_index(any_val) == Int32(LibTVMFFI.kTVMFFIDLTensorPtr)
    @test raw_data(any_val).data != 0
end

@testset "DLTensorHolder - Unified CPU/GPU" begin
    # Test that DLTensorHolder works for both CPU and GPU arrays
    # (GPU test only runs if GPU available, but we test the type unification)

    cpu_arr = Float32[1, 2, 3]
    cpu_holder = DLTensorHolder(cpu_arr)

    @test cpu_holder isa DLTensorHolder
    @test cpu_holder.tensor.device.device_type == Int32(LibTVMFFI.kDLCPU)

    # GPU arrays would create the same DLTensorHolder type
    # just with different device_type in the tensor
end
