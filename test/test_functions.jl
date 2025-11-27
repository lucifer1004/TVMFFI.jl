# Tests for Function Registration and Calling

@testset "Global Function Lookup" begin
    # Test non-existent function
    func = get_global_func("this_function_definitely_does_not_exist_xyz123")
    # Should return nothing for non-existent functions
    @test func === nothing
end

@testset "TVMFunction Type System" begin
    # Test type_index and type_key for TVMFunction
    @test type_index(TVMFunction) == Int32(LibTVMFFI.kTVMFFIFunction)
    @test type_key(TVMFunction) == "ffi.Function"

    # Test instance type_index
    echo_func = get_global_func("testing.echo")
    if echo_func !== nothing
        @test type_index(echo_func) == Int32(LibTVMFFI.kTVMFFIFunction)
    end
end

@testset "TVMFunction Display" begin
    func = get_global_func("testing.echo")
    if func !== nothing
        output = sprint(show, func)
        @test occursin("TVMFunction", output)
        @test occursin("@", output)  # Should show address
    end
end

@testset "Function Registration" begin
    # Test basic function registration
    function test_add(x::Int64, y::Int64)
        return x + y
    end

    register_global_func("julia.test.add", test_add)
    func = get_global_func("julia.test.add")
    @test func !== nothing
    @test func(Int64(10), Int64(20)) == 30

    # Test function with multiple types
    function test_multiply(x::Int64, y::Float64)
        return Float64(x) * y
    end

    register_global_func("julia.test.multiply", test_multiply)
    func2 = get_global_func("julia.test.multiply")
    result = func2(Int64(5), 3.0)
    @test result ≈ 15.0

    # Test function with boolean
    function test_negate(b::Bool)
        return !b
    end

    register_global_func("julia.test.negate", test_negate)
    func3 = get_global_func("julia.test.negate")
    @test func3(true) == false
    @test func3(false) == true

    # Test function returning nothing
    function test_void()
        return nothing
    end

    register_global_func("julia.test.void", test_void)
    func4 = get_global_func("julia.test.void")
    @test func4() === nothing

    # Test varargs function
    function test_sum(args...)
        return sum(args)
    end

    register_global_func("julia.test.sum", test_sum; override = true)
    func5 = get_global_func("julia.test.sum")
    @test func5(Int64(1), Int64(2), Int64(3), Int64(4)) == 10

    # Test exception handling
    function test_error()
        error("Test error!")
    end

    register_global_func("julia.test.error", test_error; override = true)
    func6 = get_global_func("julia.test.error")
    @test_throws TVMError func6()
end

@testset "DLTensor Stride Handling in Callbacks" begin
    # Test array callback with different stride patterns

    # Helper to register and test a callback
    function test_array_callback(arr, expected_result)
        callback_called = Ref(false)
        result_ref = Ref{Any}(nothing)

        function array_processor(x::AbstractArray)
            callback_called[] = true
            result_ref[] = copy(x)
            return x
        end

        register_global_func("julia.test.array_proc", array_processor; override = true)
        func = get_global_func("julia.test.array_proc")

        returned = func(arr)

        @test callback_called[]
        @test result_ref[] ≈ expected_result
        @test returned ≈ expected_result
    end

    # Test 1: Contiguous array (NULL strides equivalent)
    contiguous_arr = Float32[1, 2, 3, 4, 5]
    test_array_callback(contiguous_arr, contiguous_arr)

    # Test 2: Contiguous slice
    vec = Float32[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    contiguous_slice = @view vec[3:7]
    test_array_callback(contiguous_slice, collect(contiguous_slice))

    # Test 3: Column slice (contiguous in column-major)
    matrix = Float32[1 2 3; 4 5 6; 7 8 9]
    col_slice = @view matrix[:, 2]
    test_array_callback(col_slice, collect(col_slice))

    # Test 4: Row slice (non-contiguous in column-major) 
    row_slice = @view matrix[2, :]
    test_array_callback(row_slice, collect(row_slice))

    # Test 5: 2D array
    matrix_2d = Float64[1.0 2.0; 3.0 4.0]
    test_array_callback(matrix_2d, matrix_2d)
end

@testset "Callback Return Types" begin
    # Test callback returning TVMTensor
    function return_tvmtensor(x)
        return TVMTensor(x)
    end

    register_global_func("julia.test.return_tvmtensor", return_tvmtensor; override = true)
    func = get_global_func("julia.test.return_tvmtensor")

    arr = Float32[1.0, 2.0, 3.0]
    result = func(arr)
    @test result ≈ arr

    # Test callback returning TensorView
    function return_tensorview(x)
        view = TensorView(x)
        return view
    end

    register_global_func("julia.test.return_tensorview", return_tensorview; override = true)
    func2 = get_global_func("julia.test.return_tensorview")

    result2 = func2(arr)
    @test result2 ≈ arr
end

@testset "Identity Optimization" begin
    # Test that returning the same array returns the original
    identity_func = get_global_func("testing.echo")
    if identity_func !== nothing
        arr = Float32[1.0, 2.0, 3.0, 4.0]
        result = identity_func(arr)
        @test result ≈ arr

        # Test with Int
        int_result = identity_func(Int64(42))
        @test int_result == 42

        # Test with String
        str_result = identity_func("hello world")
        @test str_result == "hello world"
    end
end

@testset "Multi-Argument Calls" begin
    nop = get_global_func("testing.nop")
    if nop !== nothing
        # Test 0 args (fallback to Vector path)
        # Note: 0-arg case uses generic path

        # Test 1-10 args (specialized methods)
        @test nop(1) === nothing
        @test nop(1, 2) === nothing
        @test nop(1, 2, 3) === nothing
        @test nop(1, 2, 3, 4) === nothing
        @test nop(1, 2, 3, 4, 5) === nothing
        @test nop(1, 2, 3, 4, 5, 6) === nothing
        @test nop(1, 2, 3, 4, 5, 6, 7) === nothing
        @test nop(1, 2, 3, 4, 5, 6, 7, 8) === nothing
        @test nop(1, 2, 3, 4, 5, 6, 7, 8, 9) === nothing
        @test nop(1, 2, 3, 4, 5, 6, 7, 8, 9, 10) === nothing

        # Test 11+ args (fallback path)
        @test nop(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11) === nothing

        # Test with arrays
        arr1 = Float32[1.0]
        arr2 = Float32[2.0]
        arr3 = Float32[3.0]
        @test nop(arr1) === nothing
        @test nop(arr1, arr2) === nothing
        @test nop(arr1, arr2, arr3) === nothing
    end
end

@testset "TensorView Arguments" begin
    nop = get_global_func("testing.nop")
    if nop !== nothing
        arr = Float32[1.0, 2.0, 3.0]
        view = TensorView(arr)

        # Test passing TensorView directly
        @test nop(view) === nothing

        # Test identity with TensorView
        identity = get_global_func("testing.echo")
        if identity !== nothing
            result = identity(view)
            @test result ≈ arr
        end
    end
end
