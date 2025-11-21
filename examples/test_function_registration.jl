using TVMFFI

println("Testing Julia function registration...")

# Define a simple Julia function
function my_add(x::Int64, y::Int64)
    println("Julia function called with x=$x, y=$y")
    return x + y
end

# Register it
println("\nRegistering 'julia.my_add'...")
register_global_func("julia.my_add", my_add)
println("Registration successful!")

# Look it up
println("\nLooking up 'julia.my_add'...")
func = get_global_func("julia.my_add")

if func === nothing
    error("Failed to retrieve registered function!")
end

println("Found function: $func")

# Call it
println("\nCalling function with (10, 20)...")
result = func(Int64(10), Int64(20))
println("Result: $result")

if result != 30
    error("Unexpected result: $result (expected 30)")
end

println("\n✅ Function registration test passed!")

