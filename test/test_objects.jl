# Tests for Object Registration

@testset "Object Registration" begin
    # Test basic type registration
    struct TestObject1
        x::Int
    end

    idx = register_object("julia.test.TestObject1", TestObject1)
    @test idx > 0

    # Verify we can look it up
    idx2 = get_type_index("julia.test.TestObject1")
    @test idx == idx2

    # Test with different parent type
    struct TestObject2
        y::Float64
    end

    idx3 = register_object("julia.test.TestObject2", TestObject2; parent_type_index = Int32(64))
    @test idx3 > 0
    @test idx3 != idx
end

@testset "Reference Counting - own parameter" begin
    # Test that constructors respect own parameter
    # This is implementation detail, but important for correctness

    # Create a TVMString to test with
    s = TVMString("test")

    # The internal handle should be valid
    @test s.data.type_index >= Int32(LibTVMFFI.kTVMFFISmallStr)
end
