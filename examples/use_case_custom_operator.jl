#!/usr/bin/env julia
"""
Use Case: Custom Operator Implementation

This example demonstrates how to implement a custom operator in Julia
and make it available to TVM-compiled models.

Scenario: You have a domain-specific operation (e.g., special mathematical
function) that's already efficiently implemented in Julia, and you want
to use it in a TVM model.
"""

using TVMFFI

println("="^70)
println("Use Case: Custom Operator with Julia Implementation")
println("="^70)

# Step 1: Define custom operator in Julia
println("\n1. Defining custom operator in Julia...")

"""
    sigmoid_custom(x::Float64) -> Float64

Custom sigmoid implementation with additional domain logic.
In practice, this could be a specialized function from a Julia package
(e.g., SpecialFunctions.jl, DifferentialEquations.jl, etc.)
"""
function sigmoid_custom(x::Float64)
    # Add some custom logic (e.g., clamping for numerical stability)
    x_clamped = clamp(x, -20.0, 20.0)
    
    # Standard sigmoid
    result = 1.0 / (1.0 + exp(-x_clamped))
    
    # Could add logging, validation, etc.
    return result
end

println("  ✓ Defined sigmoid_custom(x)")

# Step 2: Register to TVM
println("\n2. Registering operator to TVM global registry...")

register_global_func("julia.ops.sigmoid_custom", sigmoid_custom)

println("  ✓ Registered as 'julia.ops.sigmoid_custom'")

# Step 3: Verify registration
println("\n3. Verifying registration...")

retrieved_func = get_global_func("julia.ops.sigmoid_custom")

if retrieved_func === nothing
    error("Failed to retrieve registered function!")
end

println("  ✓ Successfully retrieved function")

# Step 4: Test the operator
println("\n4. Testing custom operator...")

test_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
println("\n  Input values: ", test_values)
println("  Results:")

for x in test_values
    result = retrieved_func(x)
    expected = sigmoid_custom(x)
    
    println("    sigmoid_custom($x) = $result")
    
    if abs(result - expected) > 1e-10
        error("Result mismatch!")
    end
end

println("\n  ✓ All results correct")

# Step 5: Critical test - Array parameters (the real use case!)
println("\n5. Testing with array parameters...")

# Define function that takes array and returns array
function apply_sigmoid_elementwise(x::Vector{Float64})
    println("  Julia received array of length $(length(x))")
    # Apply sigmoid to each element
    return [sigmoid_custom(xi) for xi in x]
end

register_global_func("julia.ops.sigmoid_array", apply_sigmoid_elementwise)

println("  ✓ Registered sigmoid_array (takes Vector{Float64})")

# Test it
array_func = get_global_func("julia.ops.sigmoid_array")
test_input = [1.0, 2.0, 3.0, 4.0, 5.0]
println("\n  Input: ", test_input)

test_output = array_func(test_input)
println("  Output: ", round.(test_output, digits=4))

# Verify correctness
expected = [sigmoid_custom(x) for x in test_input]
if test_output ≈ expected
    println("  ✓ Array processing works correctly!")
else
    error("Output mismatch!")
end

println("\n" * "="^70)
println("✅ Custom Operator Use Case - COMPLETE")
println("="^70)

println("\n📝 Summary:")
println("  • Defined custom operator in Julia")
println("  • Registered it to TVM global registry")
println("  • Verified it's callable from TVM")
println("  • Tested with scalar and batch inputs")
println()
println("💡 Real-World Applications:")
println("  • Use SpecialFunctions.jl for mathematical operators")
println("  • Use DifferentialEquations.jl for physics simulations")
println("  • Use Optim.jl for optimization-based layers")
println("  • Use any Julia package in your TVM workflow!")
println()
println("🎯 Key Benefit: Stay in Julia, leverage entire ecosystem!")

