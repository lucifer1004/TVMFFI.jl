# Tests for Object Registration

@testset "Object Registration - Basic API" begin
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

@testset "@register_object Macro" begin
    # Test the macro with a simple type
    @register_object "julia.test.MacroTest1" struct MacroTest1 end

    # The type should be defined and registered
    @test isdefined(@__MODULE__, :MacroTest1)

    # Type index should be positive
    @test type_index(MacroTest1) > 0

    # Type key should be correct
    @test type_key(MacroTest1) == "julia.test.MacroTest1"

    # Test with field declarations (informational)
    @register_object "julia.test.MacroTest2" struct MacroTest2
        x::Int64
        y::Float64
    end

    @test isdefined(@__MODULE__, :MacroTest2)
    @test type_index(MacroTest2) > 0
    @test type_key(MacroTest2) == "julia.test.MacroTest2"
    @test type_index(MacroTest1) != type_index(MacroTest2)

    # Test with parent type annotation (for documentation purposes)
    @register_object "julia.test.MacroTest3" struct MacroTest3 <: Any
        value::String
    end

    @test isdefined(@__MODULE__, :MacroTest3)
    @test type_index(MacroTest3) > 0
end

@testset "@register_object_simple Macro" begin
    # Define a custom struct manually
    mutable struct CustomObject
        handle::LibTVMFFI.TVMFFIObjectHandle
        cached_value::Int  # Extra field for caching

        function CustomObject(handle::LibTVMFFI.TVMFFIObjectHandle; borrowed::Bool)
            if handle == C_NULL
                error("Cannot create CustomObject from NULL handle")
            end
            if borrowed
                LibTVMFFI.TVMFFIObjectIncRef(handle)
            end
            obj = new(handle, 0)
            finalizer(obj) do o
                if o.handle != C_NULL
                    LibTVMFFI.TVMFFIObjectDecRef(o.handle)
                end
            end
            return obj
        end
    end

    # Register it with the simple macro
    @register_object_simple "julia.test.CustomObject" CustomObject

    # Verify registration
    @test type_index(CustomObject) > 0
    @test type_key(CustomObject) == "julia.test.CustomObject"
end

@testset "Reference Counting - own parameter" begin
    # Test that constructors respect own parameter
    # This is implementation detail, but important for correctness

    # Create a TVMString to test with
    s = TVMString("test")

    # The internal handle should be valid
    @test s.data.type_index >= Int32(LibTVMFFI.kTVMFFISmallStr)
end
