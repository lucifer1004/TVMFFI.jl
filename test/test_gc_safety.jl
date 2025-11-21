# GC Safety Stress Tests

@testset "GC Safety Stress Test" begin
    # Test that references survive aggressive GC

    # Test 1: String creation under GC pressure
    all_strings_valid = true
    for i in 1:100
        s = TVMString("test string $i")
        # Create garbage
        _ = [rand(100) for _ in 1:10]
        GC.gc()
        # String should still be valid
        if String(s) != "test string $i"
            all_strings_valid = false
            break
        end
    end
    @test all_strings_valid

    # Test 2: Function registration under GC
    all_funcs_work = true
    for i in 1:50
        func_name = "julia.gc_test_$i"
        test_func = x -> x + i
        register_global_func(func_name, test_func; override = true)

        # Trigger GC
        _ = [rand(1000) for _ in 1:20]
        GC.gc()

        # Retrieve and call
        retrieved = get_global_func(func_name)
        if retrieved === nothing || retrieved(Int64(10)) != 10 + i
            all_funcs_work = false
            break
        end
    end
    @test all_funcs_work

    # Test 3: Object references under GC
    all_modules_valid = true
    for i in 1:50
        mod = system_lib()
        # Allocate garbage
        _ = [TVMString("garbage $j") for j in 1:100]
        GC.gc()
        # Module should still be valid
        if !(mod isa TVMModule) || get_module_kind(mod) != "library"
            all_modules_valid = false
            break
        end
    end
    @test all_modules_valid
end
