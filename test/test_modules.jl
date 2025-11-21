# Tests for Module API

@testset "Module API" begin
    # Test that module API functions are cached
    @test TVMFFI._module_loader[] !== nothing
    @test TVMFFI._function_getter[] !== nothing

    # Test exports
    @test isdefined(TVMFFI, :load_module)
    @test isdefined(TVMFFI, :get_function)
    @test isdefined(TVMFFI, :TVMModule)
    @test isdefined(TVMFFI, :system_lib)
    @test isdefined(TVMFFI, :write_to_file)
    @test isdefined(TVMFFI, :inspect_source)
    @test isdefined(TVMFFI, :get_module_kind)
    @test isdefined(TVMFFI, :implements_function)
end

@testset "Module Enhancements" begin
    # Test system_lib
    mod = system_lib()
    @test mod isa TVMModule

    # Test get_module_kind
    kind = get_module_kind(mod)
    @test kind isa String
    @test kind == "library"

    # Test implements_function
    # System lib may or may not have functions, just test the API works
    result = implements_function(mod, "nonexistent_function_12345", false)
    @test result isa Bool
end
