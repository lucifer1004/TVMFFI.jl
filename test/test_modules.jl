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

    # Test system_lib with prefix
    mod_prefix = system_lib("test_prefix")
    @test mod_prefix isa TVMModule

    # Test get_module_kind
    kind = get_module_kind(mod)
    @test kind isa String
    @test kind == "library"

    # Test implements_function
    # System lib may or may not have functions, just test the API works
    result = implements_function(mod, "nonexistent_function_12345", false)
    @test result isa Bool

    # Test get_function returns nothing for nonexistent function
    func = get_function(mod, "nonexistent_function_12345", true)
    @test func === nothing
end

@testset "TVMModule Type System" begin
    # Test type_index and type_key for TVMModule
    @test type_index(TVMModule) == Int32(LibTVMFFI.kTVMFFIModule)
    @test type_key(TVMModule) == "ffi.Module"

    # Test instance type_index
    mod = system_lib()
    @test type_index(mod) == Int32(LibTVMFFI.kTVMFFIModule)
end

@testset "TVMModule Display" begin
    mod = system_lib()
    
    # Test show
    output = sprint(show, mod)
    @test occursin("TVMModule", output)
    @test occursin("@", output)  # Should show address
end

@testset "TVMModule getindex" begin
    mod = system_lib()
    
    # Test getindex error for nonexistent function
    @test_throws ErrorException mod["nonexistent_function_12345"]
end

@testset "Module Inspection Functions" begin
    mod = system_lib()
    
    # Test inspect_source - may or may not return empty string depending on module type
    # For system_lib (library type), this typically returns empty string
    source = inspect_source(mod, "")
    @test source isa String
    
    # Test inspect_source with format
    source_ll = inspect_source(mod, "ll")  # LLVM IR format
    @test source_ll isa String
end
