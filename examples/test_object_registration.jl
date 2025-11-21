using TVMFFI

# Define a dummy Julia type
struct MyObject
    x::Int
end

# Register it
# Note: In a real scenario, we would need to register this in C++ first or use a mechanism to create a new type index dynamically.
# For now, we will try to look up an existing type index to verify the mechanism works.
# "runtime.String" is usually available.

println("Registering new type test.MyObject...")
idx = register_object("test.MyObject", MyObject)
println("Registered test.MyObject with index: $idx")

println("Looking up type index for test.MyObject...")
idx2 = get_type_index("test.MyObject")
println("Retrieved index: $idx2")

if idx != idx2
    error("Index mismatch: $idx != $idx2")
end

println("Success!")
