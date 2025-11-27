# Tests for Error Handling

@testset "Error Handling" begin
    # Test error creation
    err = TVMError(ValueError, "test message", "test backtrace")
    @test err.kind == "ValueError"
    @test err.message == "test message"
    @test err.backtrace == "test backtrace"

    # Test error kinds
    @test ValueError.name == "ValueError"
    @test TVMFFI.TypeError.name == "TypeError"  # Qualify to avoid Base.TypeError
    @test RuntimeError.name == "RuntimeError"

    # Test exception throwing
    @test_throws TVMError begin
        err = TVMError(ValueError, "thrown error", "")
        throw(err)
    end

    # Test type_index and type_key
    @test type_index(TVMError) == Int32(LibTVMFFI.kTVMFFIError)
    @test type_key(TVMError) == "ffi.Error"
    @test type_index(err) == Int32(LibTVMFFI.kTVMFFIError)
end

@testset "TVMErrorKind" begin
    # Test string conversion
    @test string(ValueError) == "ValueError"
    @test string(RuntimeError) == "RuntimeError"
    @test string(TVMFFI.TypeError) == "TypeError"
    @test string(AttributeError) == "AttributeError"
    @test string(TVMFFI.KeyError) == "KeyError"  # Qualify to avoid Base.KeyError
    @test string(TVMFFI.IndexError) == "IndexError"  # Qualify to avoid Base.IndexError
end

@testset "TVMError Display" begin
    # Test show with backtrace
    err_with_bt = TVMError(RuntimeError, "Something went wrong", "  at line 1\n  at line 2")
    output = sprint(show, err_with_bt)
    @test occursin("TVMError", output)
    @test occursin("RuntimeError", output)
    @test occursin("Something went wrong", output)
    @test occursin("Backtrace", output)

    # Test show without backtrace
    err_no_bt = TVMError(ValueError, "No backtrace here", "")
    output_no_bt = sprint(show, err_no_bt)
    @test occursin("TVMError", output_no_bt)
    @test occursin("ValueError", output_no_bt)
    @test occursin("No backtrace here", output_no_bt)

    # Test showerror
    err_output = sprint(showerror, err_with_bt)
    @test occursin("TVMError", err_output)
    @test occursin("RuntimeError", err_output)
end

@testset "check_call Error Paths" begin
    # Test check_call with 0 (success)
    @test TVMFFI.check_call(0) === nothing

    # Note: get_global_func returns nothing for non-existent functions (doesn't throw).
    # Actual check_call errors are tested implicitly through other tests
    # (e.g., function registration and callback tests that trigger real FFI errors).
end
