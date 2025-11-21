using TVMFFI

# Define a Julia function
function my_add(x::Int64, y::Int64)
    println("Called my_add with $x, $y")
    return x + y
end

# Register it as a global TVM function
println("Registering global function...")
register_global_func("test.my_add", my_add)

# Retrieve it back using get_global_func
println("Retrieving global function...")
func = get_global_func("test.my_add")

if func === nothing
    error("Failed to retrieve registered function")
end

# Call it
println("Calling function...")
result = func(10, 20)
println("Result: $result")

if result != 30
    error("Wrong result: expected 30, got $result")
end

println("Success!")
