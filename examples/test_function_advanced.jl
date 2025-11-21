using TVMFFI

println("Testing advanced function registration...")

# Test 1: Function with multiple types
function my_multiply(x::Int64, y::Float64)
    println("  Julia: my_multiply($x, $y)")
    return Float64(x) * y
end

register_global_func("julia.my_multiply", my_multiply)
func1 = get_global_func("julia.my_multiply")
result1 = func1(Int64(5), 3.14)
println("Test 1 - Mixed types: $result1 (expected ~15.7)")
@assert abs(result1 - 15.7) < 0.1

# Test 2: Function with boolean
function my_negate(b::Bool)
    println("  Julia: my_negate($b)")
    return !b
end

register_global_func("julia.my_negate", my_negate)
func2 = get_global_func("julia.my_negate")
result2 = func2(true)
println("Test 2 - Boolean: $result2 (expected false)")
@assert result2 == false

# Test 3: Function returning nothing
function my_void()
    println("  Julia: my_void() called")
    return nothing
end

register_global_func("julia.my_void", my_void)
func3 = get_global_func("julia.my_void")
result3 = func3()
println("Test 3 - Void return: $result3 (expected nothing)")
@assert result3 === nothing

# Test 4: Function with varargs (multiple params)
function my_sum(args...)
    println("  Julia: my_sum with $(length(args)) args")
    return sum(args)
end

register_global_func("julia.my_sum", my_sum; override=true)
func4 = get_global_func("julia.my_sum")
result4 = func4(Int64(1), Int64(2), Int64(3), Int64(4))
println("Test 4 - Varargs: $result4 (expected 10)")
@assert result4 == 10

# Test 5: Exception handling
function my_error()
    println("  Julia: my_error() - about to throw")
    error("This is a test error!")
end

register_global_func("julia.my_error", my_error; override=true)
func5 = get_global_func("julia.my_error")

println("\nTest 5 - Exception handling:")
try
    func5()
    error("Should have thrown an exception!")
catch e
    println("  Caught exception: $(typeof(e))")
    println("  Message preview: $(first(string(e), 50))...")
    @assert e isa TVMError
end

println("\n✅ All advanced tests passed!")

