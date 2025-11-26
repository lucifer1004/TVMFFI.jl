# Tests for Compiled Test Fixtures

# Check if CUDA is available (both hardware and CUDA.jl)
const CUDA_AVAILABLE = begin
    cuda_functional = try
        using CUDA
        functional = CUDA.functional()
        if functional
            ensure_fixture_built("add_one_cuda")
        end
        functional
    catch e
        false
    end
end

# Check if Metal is available (both hardware and Metal.jl)
const METAL_AVAILABLE = begin
    metal_functional = try
        using Metal
        functional = Metal.functional()
        if functional
            ensure_fixture_built("add_one_metal")
        end
        functional
    catch e
        false
    end
end

@testset "Test Fixtures" begin
    @testset "add_one_cpu - Basic" begin
        # Load fixture (auto-builds if needed)
        mod = load_fixture("add_one_cpu")
        @test mod isa TVMModule
        @test implements_function(mod, "add_one_cpu")

        add_one = mod["add_one_cpu"]
        @test add_one isa TVMFunction

        # Test: [1, 2, 3, 4, 5] + 1 = [2, 3, 4, 5, 6]
        x = Float32[1, 2, 3, 4, 5]
        y = similar(x)
        add_one(DLTensorHolder(x), DLTensorHolder(y))
        @test y ≈ Float32[2, 3, 4, 5, 6]

        # Test: [10, 20, 30] + 1 = [11, 21, 31]
        x2 = Float32[10, 20, 30]
        y2 = similar(x2)
        add_one(DLTensorHolder(x2), DLTensorHolder(y2))
        @test y2 ≈ Float32[11, 21, 31]

        # Test: zeros + 1 = ones
        x3 = zeros(Float32, 10)
        y3 = similar(x3)
        add_one(DLTensorHolder(x3), DLTensorHolder(y3))
        @test y3 ≈ ones(Float32, 10)
    end

    @testset "add_one_cpu - Strided Arrays" begin
        mod = load_fixture("add_one_cpu")
        add_one = mod["add_one_cpu"]

        # Test 1: 1D contiguous slice
        vec = Float32[10, 20, 30, 40, 50, 60]
        x_slice = @view vec[2:5]
        y_slice = similar(x_slice)
        add_one(DLTensorHolder(x_slice), DLTensorHolder(y_slice))
        @test y_slice ≈ Float32[21, 31, 41, 51]

        # Test 2: 1D strided view (every 2nd element)
        x_vec = Float32[1, 2, 3, 4, 5, 6, 7, 8]
        y_vec = zeros(Float32, 8)
        x_strided = @view x_vec[1:2:end]  # [1, 3, 5, 7]
        y_strided = @view y_vec[1:2:end]  # Output with stride=2
        add_one(DLTensorHolder(x_strided), DLTensorHolder(y_strided))
        @test y_strided ≈ Float32[2, 4, 6, 8]

        # Test 3: 2D contiguous array
        x_mat = Float32[1 2 3; 4 5 6]  # 2×3
        y_mat = similar(x_mat)
        add_one(DLTensorHolder(x_mat), DLTensorHolder(y_mat))
        @test y_mat ≈ Float32[2 3 4; 5 6 7]

        # Test 4: 2D column slice (contiguous in column-major)
        mat = Float32[1 2 3 4; 5 6 7 8; 9 10 11 12]  # 3×4
        x_col = @view mat[:, 2]  # [2, 6, 10]
        y_col = similar(x_col)
        add_one(DLTensorHolder(x_col), DLTensorHolder(y_col))
        @test y_col ≈ Float32[3, 7, 11]

        # Test 5: 2D row slice (NON-contiguous, stride > 1)
        x_row = @view mat[2, :]  # [5, 6, 7, 8]
        y_row = similar(x_row)
        add_one(DLTensorHolder(x_row), DLTensorHolder(y_row))
        @test y_row ≈ Float32[6, 7, 8, 9]

        # Test 6: 2D sub-matrix
        big_mat = Float32[1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]  # 4×4
        x_sub = @view big_mat[2:3, 2:3]  # 2×2 sub-matrix [[6, 7], [10, 11]]
        y_sub = similar(x_sub)
        add_one(DLTensorHolder(x_sub), DLTensorHolder(y_sub))
        @test y_sub ≈ Float32[7 8; 11 12]

        # Test 7: 3D array
        x_3d = reshape(Float32[1:24...], 2, 3, 4)  # 2×3×4
        y_3d = similar(x_3d)
        add_one(DLTensorHolder(x_3d), DLTensorHolder(y_3d))
        @test y_3d ≈ x_3d .+ 1

        # Test 8: 3D slice
        arr_3d = reshape(Float32[1:60...], 3, 4, 5)  # 3×4×5
        x_3d_slice = @view arr_3d[2:3, 2:3, 2:4]  # 2×2×3 sub-tensor
        y_3d_slice = similar(x_3d_slice)
        add_one(DLTensorHolder(x_3d_slice), DLTensorHolder(y_3d_slice))
        @test y_3d_slice ≈ x_3d_slice .+ 1

        # Test 9: Complex strided view
        complex_mat = Float32[i + j for i in 1:6, j in 1:8]  # 6×8
        x_complex = @view complex_mat[1:2:end, 1:3:end]  # Every 2nd row, 3rd col
        y_complex = similar(x_complex)
        add_one(DLTensorHolder(x_complex), DLTensorHolder(y_complex))
        @test y_complex ≈ x_complex .+ 1
    end

    @testset "add_one_cuda - GPU Arrays (Optional)" begin
        if !CUDA_AVAILABLE
            @info "CUDA tests skipped: CUDA not available"
            @test_skip "CUDA not available"
        else
            # Load CUDA fixture
            mod = load_fixture("add_one_cuda")
            @test mod isa TVMModule
            @test implements_function(mod, "add_one_cuda")

            add_one_cuda = mod["add_one_cuda"]
            @test add_one_cuda isa TVMFunction

            # Test 1: Basic CUDA array
            x_gpu = CUDA.CuArray(Float32[1, 2, 3, 4, 5])
            y_gpu = CUDA.similar(x_gpu)
            add_one_cuda(DLTensorHolder(x_gpu), DLTensorHolder(y_gpu))
            @test Array(y_gpu) ≈ Float32[2, 3, 4, 5, 6]

            # Test 2: CUDA array with stride (every 2nd element)
            x_vec_gpu = CUDA.CuArray(Float32[1, 2, 3, 4, 5, 6, 7, 8])
            y_vec_gpu = CUDA.zeros(Float32, 8)
            x_strided = @view x_vec_gpu[1:2:end]
            y_strided = @view y_vec_gpu[1:2:end]
            add_one_cuda(DLTensorHolder(x_strided), DLTensorHolder(y_strided))
            @test Array(y_strided) ≈ Float32[2, 4, 6, 8]

            # Test 3: 2D CUDA array
            x_mat_gpu = CUDA.CuArray(Float32[1 2 3; 4 5 6])
            y_mat_gpu = CUDA.similar(x_mat_gpu)
            add_one_cuda(DLTensorHolder(x_mat_gpu), DLTensorHolder(y_mat_gpu))
            @test Array(y_mat_gpu) ≈ Float32[2 3 4; 5 6 7]

            # Test 4: Column slice (contiguous on GPU)
            mat_gpu = CUDA.CuArray(Float32[1 2 3 4; 5 6 7 8; 9 10 11 12])
            x_col = @view mat_gpu[:, 2]
            y_col = CUDA.similar(x_col)
            add_one_cuda(DLTensorHolder(x_col), DLTensorHolder(y_col))
            @test Array(y_col) ≈ Float32[3, 7, 11]
        end
    end

    @testset "add_one_metal - GPU Arrays (Optional)" begin
        if !METAL_AVAILABLE
            @info "Metal tests skipped: Metal not available"
            @test_skip "Metal not available"
        else
            # Load Metal fixture
            mod = load_fixture("add_one_metal")
            @test mod isa TVMModule
            @test implements_function(mod, "add_one_metal")

            add_one_metal = mod["add_one_metal"]
            @test add_one_metal isa TVMFunction

            # Test 1: Simple 1D vector
            x_metal = Metal.MtlArray(Float32[1, 2, 3, 4, 5])
            y_metal = Metal.zeros(Float32, 5)
            add_one_metal(DLTensorHolder(x_metal), DLTensorHolder(y_metal))
            Metal.synchronize()
            @test Array(y_metal) ≈ Float32[2, 3, 4, 5, 6]

            # Test 2: 1D strided view (every 2nd element)
            x_vec_metal = Metal.MtlArray(Float32[1, 2, 3, 4, 5, 6, 7, 8])
            y_vec_metal = Metal.zeros(Float32, 8)
            x_strided = @view x_vec_metal[1:2:end]  # [1, 3, 5, 7]
            y_strided = @view y_vec_metal[1:2:end]
            add_one_metal(DLTensorHolder(x_strided), DLTensorHolder(y_strided))
            Metal.synchronize()
            @test Array(y_strided) ≈ Float32[2, 4, 6, 8]

            # Test 3: 2D Matrix (Column-Major Layout)
            x_mat_metal = Metal.MtlArray(Float32[1 2 3; 4 5 6])  # 2×3
            y_mat_metal = Metal.similar(x_mat_metal)
            add_one_metal(DLTensorHolder(x_mat_metal), DLTensorHolder(y_mat_metal))
            Metal.synchronize()
            @test Array(y_mat_metal) ≈ Float32[2 3 4; 5 6 7]

            # Test 4: Column slice (contiguous in column-major)
            mat_metal = Metal.MtlArray(Float32[1 2 3 4; 5 6 7 8; 9 10 11 12])
            x_col = @view mat_metal[:, 2]  # [2, 6, 10]
            y_col = Metal.similar(x_col)
            add_one_metal(DLTensorHolder(x_col), DLTensorHolder(y_col))
            Metal.synchronize()
            @test Array(y_col) ≈ Float32[3, 7, 11]

            # Test 5: Row slice (NON-contiguous, stride > 1)
            x_row = @view mat_metal[2, :]  # [5, 6, 7, 8]
            y_row = Metal.similar(x_row)
            add_one_metal(DLTensorHolder(x_row), DLTensorHolder(y_row))
            Metal.synchronize()
            @test Array(y_row) ≈ Float32[6, 7, 8, 9]

            # Test 6: 2D sub-matrix (complex strides)
            big_mat = Metal.MtlArray(Float32[1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16])
            x_sub = @view big_mat[2:3, 2:3]  # 2×2 sub-matrix
            y_sub = Metal.similar(x_sub)
            add_one_metal(DLTensorHolder(x_sub), DLTensorHolder(y_sub))
            Metal.synchronize()
            @test Array(y_sub) ≈ Float32[7 8; 11 12]

            @info "✓ Metal tests passed"
        end
    end
end
