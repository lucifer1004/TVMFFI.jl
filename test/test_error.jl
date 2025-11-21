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
end
