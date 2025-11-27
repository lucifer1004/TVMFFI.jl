# Tests for DLTensor and Array Handling

@testset "TensorView - CPU Arrays" begin
    # Test creating view from regular array
    x = Float32[1, 2, 3, 4, 5]
    view = TensorView(x)

    @test view isa TensorView
    @test size(view) == (5,)  # Returns NTuple
    @test view._strides == (1,)  # Internal field is NTuple
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
    @test size(view) == (5,)
    @test view._strides == (1,)  # Contiguous slice
    @test view.source === slice

    # Test column slice (contiguous in column-major)
    matrix = Float32[1 2 3; 4 5 6; 7 8 9]
    col = @view matrix[:, 2]
    col_view = TensorView(col)

    @test size(col_view) == (3,)
    @test col_view._strides == (1,)  # Contiguous

    # Test row slice (non-contiguous in column-major)
    row = @view matrix[2, :]
    row_view = TensorView(row)

    @test size(row_view) == (3,)
    @test row_view._strides == (3,)  # Non-contiguous
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

@testset "TVMTensor Type System" begin
    # Test type_index and type_key for TVMTensor
    @test type_index(TVMTensor) == Int32(LibTVMFFI.kTVMFFITensor)
    @test type_key(TVMTensor) == "ffi.Tensor"
end

@testset "GPU Support API" begin
    # Test dldevice for CPU arrays (default)
    cpu_arr = Float32[1, 2, 3]
    dev = dldevice(cpu_arr)
    @test dev.device_type == Int32(LibTVMFFI.kDLCPU)
    @test dev.device_id == 0

    # Test with 2D array
    cpu_arr2d = rand(Float32, 3, 4)
    dev2 = dldevice(cpu_arr2d)
    @test dev2.device_type == Int32(LibTVMFFI.kDLCPU)

    # Test supports_gpu_backend (all should return false without GPU packages)
    @test supports_gpu_backend(:CUDA) isa Bool
    @test supports_gpu_backend(:ROCm) isa Bool
    @test supports_gpu_backend(:Metal) isa Bool
    @test supports_gpu_backend(:oneAPI) isa Bool

    # Test with alternative name
    @test supports_gpu_backend(:AMDGPU) isa Bool

    # Test unknown backend
    @test supports_gpu_backend(:UnknownBackend) == false

    # Test list_available_gpu_backends
    backends = list_available_gpu_backends()
    @test backends isa Vector{Symbol}
    # Without GPU packages loaded, should be empty
    # (unless running on a system with GPU packages)
end

@testset "GPU Info Functions" begin
    # Test print_gpu_info (just verify it runs without error)
    # Note: print_gpu_info() doesn't accept IO parameter, so can't use sprint
    @test_nowarn print_gpu_info()

    # Test gpu_array_info with CPU array (diagnostic function, just verify no errors)
    cpu_arr = Float32[1, 2, 3]
    @test_nowarn gpu_array_info(cpu_arr)
end

@testset "TensorView - Advanced" begin
    # Test size, ndims, length for TensorView
    arr = rand(Float32, 3, 4, 5)
    view = TensorView(arr)

    @test size(view) == (3, 4, 5)
    @test ndims(view) == 3
    @test length(view) == 60

    # Test device and dtype
    @test device(view).device_type == Int32(LibTVMFFI.kDLCPU)
    @test dtype(view).bits == 32

    # Test contiguity checks
    @test TVMFFI.is_contiguous(view) == true
    @test TVMFFI.is_f_contiguous(view) == true  # Julia arrays are column-major
    @test TVMFFI.is_c_contiguous(view) == false  # Not row-major

    # Test ownership
    @test TVMFFI.is_julia_owned(view) == true
    @test TVMFFI.is_tvm_owned(view) == false
end

@testset "TensorView - Strided Arrays" begin
    # Test non-contiguous array (row slice)
    matrix = rand(Float32, 4, 5)
    row = @view matrix[2, :]  # Non-contiguous in column-major

    row_view = TensorView(row)
    @test size(row_view) == (5,)
    @test row_view._strides == (4,)  # Stride of 4 between elements

    # Test column slice (contiguous)
    col = @view matrix[:, 3]
    col_view = TensorView(col)
    @test size(col_view) == (4,)
    @test col_view._strides == (1,)  # Contiguous

    # Test 2D subarray
    sub = @view matrix[2:3, 2:4]
    sub_view = TensorView(sub)
    @test size(sub_view) == (2, 3)
end

@testset "TensorView - Various dtypes" begin
    # Test different element types
    for T in [Float16, Float32, Float64, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64]
        arr = T[1, 2, 3]
        view = TensorView(arr)
        @test eltype(view) == T
        @test TVMFFI.source_type(view) == typeof(arr)
    end
end

@testset "Stride Computation" begin
    # Test C-contiguous strides (row-major) - Vector
    @test TVMFFI._compute_c_contiguous_strides([3, 4, 5]) == [20, 5, 1]
    @test TVMFFI._compute_c_contiguous_strides([10]) == [1]
    @test TVMFFI._compute_c_contiguous_strides(Int64[]) == Int64[]

    # Test C-contiguous strides - Tuple (zero-alloc)
    @test TVMFFI._compute_c_contiguous_strides((3, 4, 5)) == (20, 5, 1)
    @test TVMFFI._compute_c_contiguous_strides((10,)) == (1,)

    # Test F-contiguous strides (column-major, Julia-style) - Vector
    @test TVMFFI._compute_f_contiguous_strides([3, 4, 5]) == [1, 3, 12]
    @test TVMFFI._compute_f_contiguous_strides([10]) == [1]
    @test TVMFFI._compute_f_contiguous_strides(Int64[]) == Int64[]

    # Test F-contiguous strides - Tuple (zero-alloc)
    @test TVMFFI._compute_f_contiguous_strides((3, 4, 5)) == (1, 3, 12)
    @test TVMFFI._compute_f_contiguous_strides((10,)) == (1,)

    # Alias should work
    @test TVMFFI._compute_contiguous_strides([3, 4]) == TVMFFI._compute_c_contiguous_strides([3, 4])
end
