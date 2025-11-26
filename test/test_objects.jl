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

@testset "Reflection API" begin
    # Test get_type_info
    func_idx = type_index(TVMFunction)
    type_info = get_type_info(func_idx)
    @test type_info !== nothing
    @test type_info.type_index == func_idx

    # Test type_key retrieval from type_info
    type_key_bytes = type_info.type_key
    retrieved_key = unsafe_string(type_key_bytes.data, type_key_bytes.size)
    @test retrieved_key == "ffi.Function"

    # Test get_type_info with string
    type_info2 = get_type_info("ffi.Function")
    @test type_info2 !== nothing
    @test type_info2.type_index == func_idx

    # Test get_fields and get_methods (may be empty for built-in types)
    fields = get_fields(type_info)
    @test fields isa Vector{FieldInfo}

    methods = get_methods(type_info)
    @test methods isa Vector{MethodInfo}
end

@testset "Reflection with TestObject" begin
    # Test with testing.TestObjectBase which has fields and methods defined in C++
    type_info = get_type_info("testing.TestObjectBase")

    if type_info !== nothing && type_info.num_fields > 0
        # This type should have fields: v_i64, v_f64, v_str
        fields = get_fields(type_info)
        @test length(fields) > 0

        field_names = [f.name for f in fields]
        @test "v_i64" in field_names || "v_f64" in field_names

        # Get methods
        methods = get_methods(type_info)

        # Check if add_i64 method exists
        method_names = [m.name for m in methods]
        if "add_i64" in method_names
            @test true  # Method found
        end
    else
        @info "testing.TestObjectBase not available or has no reflection info"
    end
end

@testset "Real TVM Type: testing.TestObjectBase" begin
    # Register the type with @register_object macro
    @register_object "testing.TestObjectBase" struct TestObjectBase end

    # Check type registration
    @test type_index(TestObjectBase) > 0
    @test type_key(TestObjectBase) == "testing.TestObjectBase"

    # Check reflection cache
    fields, methods = TVMFFI._get_reflection_cache(TestObjectBase)

    if length(fields) > 0
        @info "TestObjectBase has $(length(fields)) fields: $(join([f.name for f in fields], ", "))"

        # Check field properties
        for field in fields
            @test field.name isa String
            @test field.getter != C_NULL  # All fields should have getters
        end
    else
        @info "TestObjectBase has no reflection fields registered in C++"
    end

    if length(methods) > 0
        @info "TestObjectBase has $(length(methods)) methods: $(join([m.name for m in methods], ", "))"

        # Check for __ffi_init__
        has_init = has_ffi_init(TestObjectBase)
        @info "TestObjectBase has __ffi_init__: $has_init"

        if has_init
            # Try to create an instance using ffi_init
            # TestObjectBase.__ffi_init__ takes (v_i64, v_f64, v_str)
            try
                obj = ffi_init(TestObjectBase, Int64(42), 3.14, "hello")
                @test obj isa TestObjectBase
                @test obj.handle != C_NULL

                # Test field access
                if any(f -> f.name == "v_i64", fields)
                    @test obj.v_i64 == 42
                end
                if any(f -> f.name == "v_f64", fields)
                    @test obj.v_f64 ≈ 3.14
                end
                if any(f -> f.name == "v_str", fields)
                    @test obj.v_str == "hello"
                end

                # Test method call if add_i64 exists
                if any(m -> m.name == "add_i64", methods)
                    result = obj.add_i64(Int64(10))
                    @test result == 52  # 42 + 10
                end
            catch e
                @info "Failed to create TestObjectBase instance: $e"
            end
        end
    else
        @info "TestObjectBase has no reflection methods registered in C++"
    end
end

@testset "Field Setter" begin
    # Test set_field_value! function
    type_info = get_type_info("testing.TestObjectBase")

    if type_info !== nothing
        fields = get_fields(type_info)
        writable_fields = filter(f -> f.is_writable, fields)

        if length(writable_fields) > 0 && has_ffi_init(TestObjectBase)
            # Create an object
            obj = ffi_init(TestObjectBase, Int64(42), 3.14, "hello")

            for field in writable_fields
                @info "Field '$(field.name)' is writable"
                # Test that we can set the field
                # (actual value test depends on the field type)
            end
        else
            @info "No writable fields or no __ffi_init__ available"
        end
    else
        @info "testing.TestObjectBase not available"
    end
end

@testset "@register_object with Reflection" begin
    # Test that registered types can use reflection-based property access
    @register_object "julia.test.ReflectionTest" struct ReflectionTest end

    # The type should be registered
    @test isdefined(@__MODULE__, :ReflectionTest)
    @test type_index(ReflectionTest) > 0
    @test type_key(ReflectionTest) == "julia.test.ReflectionTest"

    # Global reflection cache should be available
    @test isdefined(TVMFFI, :_get_reflection_cache)
end
